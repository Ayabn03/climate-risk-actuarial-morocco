# Integration of Climate Risks into Actuarial Models: Application to the Moroccan Non-Life Insurance Market

This repository contains the Python code and data processing scripts used in my Master’s thesis in Actuarial Science and Financial Markets (*Université Hassan II de Casablanca, FSJES Aïn Sebaâ*).

## Project Overview
Traditional actuarial models rely heavily on the stationarity assumption of historical loss distributions, making them inadequate for capturing non-stationary climate risks. This project proposes a data-driven framework combining:
- **Climate Reanalysis Data (ERA5 / Copernicus)** to extract climate indices (SPI-12, TXx, R95p, FFDI).
- **Sectoral Insurance Statistics (ACAPS)** covering non-life loss ratios ($S/P$) from 2016 to 2023.
- **Disaster Database (EM-DAT)** to quantify the Moroccan climate protection gap.
- **Machine Learning & GLM Models** (Linear Regression, Random Forest, XGBoost) to model non-life loss ratios.
- **NGFS Climate Scenarios** for long-term stress testing up to 2100.

---

## Repository Structure
```text
├── data/
│   ├── indices_climatiques_maroc.csv    # Processed ERA5 climate indices
│   └── base_finale_modelisation.csv     # Merged dataset (ACAPS + Climate)
├── scripts/
│   ├── 01_era5_climate_indices.py       # Extraction & calculation of climate indices (Annexe A)
│   └── 02_actuarial_modeling.py         # Model training, comparison & stress tests (Annexe B)
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
