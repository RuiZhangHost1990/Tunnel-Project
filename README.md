# Tunnel Settlement Prediction Using Deep Learning

## Overview

This repository contains a Jupyter Notebook for data-driven prediction of tunnelling-induced ground settlement. The currently available notebook, `CNNPATCHTsT-checkpoint.ipynb`, implements a sequence-to-sequence regression workflow using historical settlement measurements, project-related input features, and a recurrent neural network (GRU).

The notebook is intended to predict settlement at 30 monitoring points over a five-step future horizon from 25 historical time steps. Static input features and a time-step variable are also incorporated into the model.

> **Implementation note:** Although the notebook filename and its associated commit message refer to CNN-PatchTST/PINN source code, the notebook code currently visible in the repository defines a GRU-based model (`SimpleRNN` and `CombinedModel`). It does not, in its present form, implement a CNN-PatchTST architecture or an explicit physics-informed neural network (PINN) loss. The repository description should be updated when the corresponding CNN-PatchTST/PINN implementation is uploaded, or the notebook should be renamed to reflect its actual contents.

## Repository Contents

| File | Description |
|---|---|
| `CNNPATCHTsT-checkpoint.ipynb` | Jupyter Notebook containing data preprocessing, GRU-based sequence modelling, model training, evaluation, and Excel export of prediction results. |
| `README.md` | Project overview, data requirements, installation instructions, and usage guidance. |
| `requirements.txt` | Python dependencies for running the notebook. |
| `LICENSE` | Licence terms for repository code, subject to the ownership and third-party-material limitations stated in the file. |

## Methodology in the Available Notebook

The notebook performs the following steps:

1. Reads settlement and project-feature data from an Excel workbook.
2. Removes rows containing missing values.
3. Extracts 20 input features, one time-step variable, and settlement measurements from 30 monitoring points.
4. Constructs samples using 25 historical time steps and a five-step prediction horizon.
5. Applies min–max scaling to the input features, time-step variable, and settlement values.
6. Trains a two-layer GRU-based model. The final hidden state is combined with projected static features and the time-step variable.
7. Evaluates predictions on the training and test subsets.
8. Exports detailed predictions and evaluation statistics to `model_results.xlsx`.

The model uses mean squared error (MSE) as its training loss and reports MAE, MSE, RMSE, and R² for each monitoring point.

## Dataset

The dataset is not included in this repository. Users must obtain the required data separately and update the input file path in the notebook.

The current notebook expects an Excel workbook with the following column layout:

- Column 1: model or sample identifier.
- Columns 2–21: 20 input features.
- Column 22: time-step variable.
- Columns 23–52: settlement measurements at 30 monitoring points.

The notebook uses 25 historical steps to predict the subsequent five steps. Each sample/model therefore needs at least 30 valid time records after the missing-value rows have been removed.

**Important:** The column positions and data structure above are taken from the current notebook implementation. They should be checked against the user's actual workbook before running the code.

## Installation

Python 3.10 or later is recommended. Create and activate a virtual environment:

```bash
python -m venv tunnel-env
```

Activate it on Windows:

```bash
tunnel-env\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source tunnel-env/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your operating system, Python version, and CUDA version. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/RuiZhangHost1990/Tunnel-Project.git
   cd Tunnel-Project
   ```

2. Install the dependencies as described above.

3. Open `CNNPATCHTsT-checkpoint.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

4. Update the Excel input path in the `main()` function. The current notebook contains a machine-specific local path, which must be replaced with the location of your own dataset.

5. Run the notebook cells in order.

The current implementation automatically selects CUDA when available; otherwise, it runs on CPU.

## Outputs

The notebook is configured to save:

- `best_model.pth`: model parameters with the lowest validation loss encountered during training.
- `model_results.xlsx`: detailed training/test predictions, monitoring-point statistics, time-series prediction records, and input-feature records.

These files are generated when the notebook is executed and are not guaranteed to be present in the repository.

## Reproducibility and Limitations

The notebook sets NumPy and PyTorch random seeds to 42. Exact reproducibility may still depend on the software environment, hardware, and CUDA configuration.

The current code uses the test data loader as the validation loader during training and selects the checkpoint with the lowest loss on that loader. Consequently, the reported test performance should not be interpreted as an independent, untouched test estimate. For a rigorous evaluation, a separate validation subset should be created from the training data, and the test set should only be used for final evaluation.

The current preprocessing fits the min–max scalers before the training/test split and includes both historical and future settlement values when fitting the settlement scaler. This can introduce information leakage. For strict evaluation, fit all preprocessing transformations on the training data only and apply the fitted transformations to validation and test data.

The available notebook should be treated as the implementation actually provided in the repository. CNN-PatchTST and PINN-specific functionality should only be claimed after the corresponding source code and physics-informed formulation have been added and verified.

## Citation

If you use this code in academic work, please cite the associated publication when it becomes available. Until then, please refer to this repository and its commit history.

## Licence

See the `LICENSE` file for the applicable terms. The licence for code does not grant rights to use or redistribute any dataset that is not included in this repository.
