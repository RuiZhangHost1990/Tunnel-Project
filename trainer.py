from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from .config import NTUConfig
from .data import NTUData
from .models import MLP, PosParam, nth_derivative, pde_residual

def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

class ExcelPINNSolverTorch:
    def __init__(self, cfg: NTUConfig, device: str | None = None):
        self.cfg = cfg
        self.out = cfg.resolved_output_dir()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float32)
        np.random.seed(cfg.random_state)
        torch.manual_seed(cfg.random_state)

        self.data = NTUData(cfg)
        self.AB = (None, None)
        self._build_tensors()
        self._build_nets()
        self.history = []

    def _build_tensors(self):
        d = self.data
        self.X_train_t = torch.tensor(d.normalized_X(d.X_train), dtype=torch.float32, device=self.device)
        self.y_train_t = torch.tensor((d.y_train - d.y_mean) / d.y_std, dtype=torch.float32, device=self.device)
        self.X_test_t = torch.tensor(d.normalized_X(d.X_test), dtype=torch.float32, device=self.device)
        self.y_mean_t = torch.tensor(d.y_mean, dtype=torch.float32, device=self.device)
        self.y_std_t = torch.tensor(d.y_std, dtype=torch.float32, device=self.device)

    def _build_nets(self):
        in_dim = self.X_train_t.shape[1]
        out_dim = self.y_train_t.shape[1]
        self.net_pde_raw = MLP(1, [64, 64, 64, 64], 1).to(self.device)
        self.net_data = MLP(in_dim, [128, 128, 128, 128], out_dim).to(self.device)
        self.alpha_param = PosParam(0.09).to(self.device)
        self.beta_param = PosParam(0.01).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.net_pde_raw.parameters()) + list(self.net_data.parameters())
            + list(self.alpha_param.parameters()) + list(self.beta_param.parameters()),
            lr=self.cfg.learning_rate,
        )

    def target_symmetric_pairs(self):
        n_y = self.data.y.shape[1]
        pairs = [(0, 1)] if n_y > 1 else []
        if n_y > 2:
            indices = list(range(2, n_y))
            if self.cfg.center_target_index is not None and self.cfg.center_target_index in indices:
                indices.remove(self.cfg.center_target_index)
            i, j = 0, len(indices) - 1
            while i < j:
                pairs.append((indices[i], indices[j]))
                i += 1
                j -= 1
        return pairs

    def sample_x(self, n_total, L_focus=200.0, tail_ratio=0.2):
        return torch.rand(n_total, 1, device=self.device) * self.cfg.l_pos

    def w_num(self, x):
        return -F.softplus(self.net_pde_raw(x))

    def _compute_AB(self):
        """Exact displacement/slope continuity at x=l for analytical Region III."""
        alpha = self.alpha_param()
        beta = self.beta_param()
        x_l = torch.tensor([[self.cfg.l_pos]], dtype=torch.float32, device=self.device, requires_grad=True)
        w_II = self.w_num(x_l)
        w_II_x = nth_derivative(w_II, x_l, 1)
        A = w_II.clone()
        B = (w_II_x + alpha * A) / beta
        return A, B

    def w_from_x(self, x):
        """Numerical Region II for x<=l and analytical bounded Region III for x>l."""
        w_num = self.w_num(x)
        A, B = self.AB
        if A is None or B is None:
            return w_num
        alpha = self.alpha_param()
        beta = self.beta_param()
        xi = x - self.cfg.l_pos
        w_ana = torch.exp(-alpha * xi) * (A * torch.cos(beta * xi) + B * torch.sin(beta * xi))
        return torch.where(x <= self.cfg.l_pos, w_num, w_ana)

    def interface_loss(self, l_pos=None, return_all=False):
        """Soft moment/shear continuity; displacement/slope are exact via A and B."""
        l_pos = self.cfg.l_pos if l_pos is None else l_pos
        alpha = self.alpha_param()
        beta = self.beta_param()
        A, B = self.AB
        if A is None or B is None:
            A, B = self._compute_AB()
            self.AB = (A, B)

        x_l = torch.tensor([[l_pos]], dtype=torch.float32, device=self.device, requires_grad=True)
        w_II = self.w_num(x_l)
        w_II_x = nth_derivative(w_II, x_l, 1)
        w_II_xx = nth_derivative(w_II, x_l, 2)
        w_II_xxx = nth_derivative(w_II, x_l, 3)

        xi = torch.zeros_like(x_l, requires_grad=True)
        w_III = torch.exp(-alpha * xi) * (A * torch.cos(beta * xi) + B * torch.sin(beta * xi))
        w_III_x = torch.autograd.grad(
            w_III, xi, grad_outputs=torch.ones_like(w_III), create_graph=True, retain_graph=True
        )[0]
        w_III_xx = nth_derivative(w_III, xi, 2)
        w_III_xxx = nth_derivative(w_III, xi, 3)
        L0 = (w_II - w_III).pow(2)
        L1 = (w_II_x - w_III_x).pow(2)
        L2 = (w_II_xx - w_III_xx).pow(2)
        L3 = (w_II_xxx - w_III_xxx).pow(2)
        total = torch.mean(L2 + L3)
        if return_all:
            return total, torch.mean(L0), torch.mean(L1), torch.mean(L2), torch.mean(L3)
        return total

    def loss_components(self):
        mse = nn.MSELoss()
        self.AB = self._compute_AB()
        x_pde = self.sample_x(self.cfg.pde_points).requires_grad_(True)
        alpha = self.alpha_param()
        beta = self.beta_param()
        res = pde_residual(self.w_from_x, x_pde, alpha, beta)
        L_pde = torch.mean(res**2)
        pred_data = self.net_data(self.X_train_t)
        L_data = mse(pred_data, self.y_train_t)
        L_interface = self.interface_loss()
        x_40 = torch.tensor([[40.0]], dtype=torch.float32, device=self.device)
        L_bc1 = torch.tensor(0.0, device=self.device)
        L_bc2 = torch.mean((self.w_from_x(x_40) - self.cfg.fem_w_40) ** 2)
        pred_denorm = pred_data * self.y_std_t + self.y_mean_t
        L_sym = torch.tensor(0.0, device=self.device)
        pairs = self.target_symmetric_pairs()
        for i, j in pairs:
            diff = torch.abs(pred_denorm[:, i] - pred_denorm[:, j]) - self.cfg.error_tolerance
            L_sym = L_sym + torch.mean(F.relu(diff) ** 2)
        if pairs:
            L_sym = L_sym / len(pairs)
        L_prior = 1e-3 * (alpha - 0.08).pow(2) + 1e-2 * (beta - 0.0).pow(2)
        return {
            "data": L_data, "pde": L_pde, "interface": L_interface,
            "bc1": L_bc1, "bc2": L_bc2, "sym": L_sym, "prior": L_prior,
            "alpha": alpha.detach(), "beta": beta.detach(),
        }

    def train(self, iterations=10000, print_every=500, weights=None, dynamic_weight=True):
        weights = weights or {}
        self.history = []
        for step in range(1, iterations + 1):
            c = self.loss_components()
            if dynamic_weight:
                t = step / iterations
                w_data = 0.9 * (1 - t) + 0.1
                w_pde = 1.0 - w_data
            else:
                w_data = float(weights.get("data", 0.5))
                w_pde = float(weights.get("pde", 0.5))
            L_total = (
                w_data * c["data"]
                + w_pde * c["pde"]
                + float(weights.get("bc", 1.0)) * (self.cfg.lambda_bc1 * c["bc1"] + self.cfg.lambda_bc2 * c["bc2"])
                + self.cfg.lambda_interface * c["interface"]
                + self.cfg.lambda_sym * c["sym"]
                + self.cfg.lambda_prior * c["prior"]
            )
            self.optimizer.zero_grad(set_to_none=True)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.net_pde_raw.parameters()) + list(self.net_data.parameters())
                + list(self.alpha_param.parameters()) + list(self.beta_param.parameters()), 5.0
            )
            self.optimizer.step()
            rec = {"step": step, "total": float(L_total.detach().cpu()), "w_data": w_data, "w_pde": w_pde}
            for k in ["data", "pde", "interface", "bc1", "bc2", "sym", "prior", "alpha", "beta"]:
                rec[k] = float(c[k].detach().cpu())
            self.history.append(rec)
            if print_every and (step == 1 or step % print_every == 0 or step == iterations):
                print(f"[{step:05d}] total={rec['total']:.3e}, data={rec['data']:.3e}, pde={rec['pde']:.3e}, interface={rec['interface']:.3e}, cond40={rec['bc2']:.3e}, sym={rec['sym']:.3e}, alpha={rec['alpha']:.4f}, beta={rec['beta']:.4f}")
        self.AB = self._compute_AB()
        Lint, L0, L1, L2, L3 = self.interface_loss(return_all=True)
        print(f"Interface condition summary (x = l = {self.cfg.l_pos:g}):")
        print(f"  L0 displacement continuity = {float(L0.detach().cpu()):.6e}")
        print(f"  L1 slope continuity        = {float(L1.detach().cpu()):.6e}")
        print(f"  L2 moment continuity       = {float(L2.detach().cpu()):.6e}")
        print(f"  L3 shear continuity        = {float(L3.detach().cpu()):.6e}")
        print(f"  Total interface loss       = {float(Lint.detach().cpu()):.6e}")
        self.save_convergence_plot()
        self.save_history()
        return self.evaluate()

    def predict(self, X):
        self.net_data.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.data.normalized_X(X), dtype=torch.float32, device=self.device)
            yn = self.net_data(Xt).cpu().numpy()
        return self.data.denormalize_y(yn)

    def evaluate(self, X=None, y=None, prefix="test"):
        if X is None:
            X = self.data.X_test
        if y is None:
            y = self.data.y_test
        y_pred = self.predict(X)
        metrics = {
            "prefix": prefix,
            "MAE": float(mean_absolute_error(y, y_pred)),
            "RMSE": rmse_np(y, y_pred),
            "R2": float(r2_score(y, y_pred)),
        }
        return metrics

    def load_checkpoint(self, filename="pinn_solver.pt"):
        path = Path(filename)
        if not path.is_absolute():
            path = self.out / "models" / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net_data.load_state_dict(checkpoint["net_data"])
        self.net_pde_raw.load_state_dict(checkpoint["net_pde_raw"])
        self.alpha_param.load_state_dict(checkpoint["alpha"])
        self.beta_param.load_state_dict(checkpoint["beta"])
        AB = checkpoint.get("AB", (None, None))
        self.AB = tuple(v.to(self.device) if v is not None else None for v in AB)
        return path

    def evaluate_excel(self, excel_path, output_filename="full_pipeline_predictions.csv", prefix="full_pipeline"):
        df = pd.read_excel(excel_path, engine="openpyxl")
        if self.cfg.skip_first_data_row:
            df = df.iloc[1:].reset_index(drop=True)
        X = df.iloc[:, self.cfg.input_start:self.cfg.input_end].values.astype(np.float32)
        y = df.iloc[:, self.cfg.target_start:].values.astype(np.float32)
        y_pred = self.predict(X)
        metrics = {
            "prefix": prefix,
            "MAE": float(mean_absolute_error(y, y_pred)),
            "RMSE": rmse_np(y, y_pred),
            "R2": float(r2_score(y, y_pred)),
        }

        rows = []
        groups = np.arange(len(X)) // self.cfg.stage2_rows_per_time_step
        rows_in_group = np.arange(len(X)) % self.cfg.stage2_rows_per_time_step
        for i in range(len(X)):
            row = {
                "sample": i,
                "time_step_group": int(groups[i]),
                "row_in_time_step": int(rows_in_group[i]),
            }
            for j, col in enumerate(df.columns[self.cfg.target_start:]):
                row[f"{col}_actual"] = y[i, j]
                row[f"{col}_pred"] = y_pred[i, j]
                row[f"{col}_abs_error"] = abs(y[i, j] - y_pred[i, j])
            rows.append(row)
        out = self.out / "tables" / output_filename
        pd.DataFrame(rows).to_csv(out, index=False)
        return metrics, out

    def physics_metrics(self):
        with torch.enable_grad():
            c = self.loss_components()
            Lint, L0, L1, L2, L3 = self.interface_loss(return_all=True)
        return {
            "L_pde": float(c["pde"].detach().cpu()),
            "L_bc": float((c["bc1"] + c["bc2"]).detach().cpu()),
            "L_sym": float(c["sym"].detach().cpu()),
            "L_interface": float(Lint.detach().cpu()),
            "L_continuity_w": float(L0.detach().cpu()),
            "L_continuity_slope": float(L1.detach().cpu()),
            "L_continuity_moment": float(L2.detach().cpu()),
            "L_continuity_shear": float(L3.detach().cpu()),
        }

    def save_history(self):
        if self.history:
            pd.DataFrame(self.history).to_csv(self.out / "tables" / "training_history.csv", index=False)

    def save_convergence_plot(self):
        if not self.history:
            return None
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.history)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(df["step"], df["total"], color="royalblue", alpha=0.35, linewidth=1.2, label="Raw Loss")
        if len(df) >= 50:
            smooth = df["total"].rolling(window=50).mean().dropna()
            ax1.plot(df["step"].iloc[-len(smooth):], smooth, color="blue", linewidth=2.3, label="Smoothed Loss")
        ax1.set_yscale("log")
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Total Loss (log scale)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(df["step"], df["alpha"], color="tomato", linewidth=2.0, label="alpha")
        ax2.plot(df["step"], df["beta"], color="seagreen", linestyle="--", linewidth=2.0, label="beta")
        ax2.set_ylabel("alpha / beta Values")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9, frameon=True)
        plt.title("PINN Training Convergence")
        plt.tight_layout()
        path = self.out / "figures" / "new202605_convergence.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_predictions(self, filename="pinn_predictions.csv"):
        y_pred = self.predict(self.data.X_test)
        rows = []
        for i in range(len(self.data.X_test)):
            sample_idx = int(self.data.test_idx[i])
            row = {
                "set": "test",
                "sample": sample_idx,
                "time_step_group": int(self.data.time_step_group[sample_idx]),
                "row_in_time_step": int(self.data.time_step_row_in_group[sample_idx]),
            }
            for j, col in enumerate(self.data.target_cols):
                row[f"{col}_actual"] = self.data.y_test[i, j]
                row[f"{col}_pred"] = y_pred[i, j]
                row[f"{col}_abs_error"] = abs(self.data.y_test[i, j] - y_pred[i, j])
            rows.append(row)
        out = self.out / "tables" / filename
        pd.DataFrame(rows).to_csv(out, index=False)
        return out

    def save_checkpoint(self, filename="pinn_solver.pt"):
        path = self.out / "models" / filename
        torch.save({
            "net_data": self.net_data.state_dict(),
            "net_pde_raw": self.net_pde_raw.state_dict(),
            "alpha": self.alpha_param.state_dict(),
            "beta": self.beta_param.state_dict(),
            "AB": tuple(v.detach().cpu() if v is not None else None for v in self.AB),
            "cfg": self.cfg.__dict__,
        }, path)
        return path
