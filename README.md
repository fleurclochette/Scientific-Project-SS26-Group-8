# Scientific Project SS26 — Group 8

**Financial Market Panic and Recovery**

This repository contains the data files, and code files for the Scientific Project: Application of AI (Group 8).

## Repository Structure

- `notebooks/` — Main analysis code, organized in numerical order from data collection through the final stress test.
- `notebooks/methodological development/` — Preliminary and validation analyses used to develop the structural-break methodology, including S&P 500 preprocessing, simulated-data validation, and bootstrap block-length sensitivity analysis with S&P 500 data.
- `src/` — Reusable source code. It includes the custom structural-break implementation.
- `requirements.txt` — Python packages required to run the analysis.

The methodological-development notebooks are retained for transparency and validation but are not part of the final analysis conducted.

## Running the Analysis

Clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd Scientific-Project-SS26
pip install -r requirements.txt
```

The main analysis is contained in the `notebooks/` directory. To reproduce the analysis, run the main notebooks in numerical order, beginning with:

`01_data_collection.ipynb`

and continuing through:

`07_unseen_asset_stress_test.ipynb`
