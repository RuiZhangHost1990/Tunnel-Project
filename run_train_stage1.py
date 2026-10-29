from pathlib import Path
import argparse
import torch
from ntu_pinn_viz.config import NTUConfig
from ntu_pinn_viz.stage1 import train_stage1_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrained_stage1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = NTUConfig(stage1_model_type="cnn_patchtst", stage1_epochs=args.epochs,
                    stage1_excel_path=str(ROOT / "stage1_data/FinalTimeSeriersAnalysisData.xlsx"),
                    output_dir=str(args.output.resolve()))
    result = train_stage1_model(cfg, device=cfg.device, checkpoint_selection="last_epoch")
    print("Stage-I checkpoint:", result["best_model_path"])
    print(result["split_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
