import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

save_path = "/content/drive/MyDrive/PFE_Data/ERA5/"
results_path = "/content/drive/MyDrive/PFE_Data/Results/"
os.makedirs(results_path, exist_ok=True)
files = sorted(glob.glob(save_path + "*.nc"))
ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
t2m = ds["t2m"].mean(dim=["latitude", "longitude"]) - 273.15
precip = ds["tp"].mean(dim=["latitude", "longitude"]) * 1000
d2m = ds["d2m"].mean(dim=["latitude", "longitude"]) - 273.15
u10 = ds["u10"].mean(dim=["latitude", "longitude"])
v10 = ds["v10"].mean(dim=["latitude", "longitude"])
wind = np.sqrt(u10**2 + v10**2)
rh = (
    100
    * (np.exp(17.625 * d2m / (243.04 + d2m)))
    / (np.exp(17.625 * t2m / (243.04 + t2m)))
)
years = pd.DatetimeIndex(ds.valid_time.values).year
results = []
for year in sorted(set(years.tolist())):
  mask = years == year
  results.append({
      "annee": year,
      "tmax_annual": float(np.nanmax(t2m.values[mask])),
      "tmean_annual": float(np.nanmean(t2m.values[mask])),
      "precip_annual": float(np.nansum(precip.values[mask])),
      "wind_annual": float(np.nanmean(wind.values[mask])),
      "rh_annual": float(np.nanmean(rh.values[mask])),
      "precip_daily": precip.values[mask].tolist(),
  })
df = pd.DataFrame(results)
ref = (df.annee > 1981) & (df.annee <= 2010)
df["TXx"] = df["tmax_annual"] - df[ref]["tmax_annual"].mean()
alpha, loc, beta = stats.gamma.fit(
    df["precip_annual"].values[ref.values], floc=0
)
df["SPI_12"] = [
    float(
        stats.norm.ppf(
            np.clip(
                stats.gamma.cdf(p, alpha, loc=loc, scale=beta), 0.001, 0.999
            )
        )
    )
    for p in df["precip_annual"]
]
p95 = np.percentile(
    [
        x
        for _, r in df[ref].iterrows()
        for x in r["precip_daily"]
        if x > 0.1
    ],
    95,
)
df["R95p"] = [
    np.nansum(np.array(r["precip_daily"])[np.array(r["precip_daily"]) > p95])
    / np.nansum(r["precip_daily"])
    * 100
    for _, r in df.iterrows()
]
df["drought_factor"] = np.clip(10 - 2.5 * df["SPI_12"], 0, 10)
df["FFDI"] = df.apply(
    lambda row: 2.0
    * np.exp(
        -0.45
        + 0.987 * np.log(max(row["drought_factor"], 0.1))
        - 0.0345 * row["rh_annual"]
        + 0.0338 * row["tmean_annual"]
        + 0.0234 * row["wind_annual"]
        - 3.6
    ),
    axis=1,
)
df.to_csv(results_path + "indices_climatiques_maroc.csv", index=False)
