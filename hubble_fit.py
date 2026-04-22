import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# speed of light (km/s)
c = 299792.458

# load data
df = pd.read_csv("galaxy_data.csv")

d = df["distance_mpc"].to_numpy()
z = df["redshift"].to_numpy()

# convert redshift to velocity
v = c * z

# linear regression
m, b = np.polyfit(d, v, 1)
H0_est = m

# predicted values and residuals
v_fit = m * d + b
residuals = v - v_fit

print(f"Estimated H0 = {H0_est:.2f} km/s/Mpc")
print(f"Intercept = {b:.2f} km/s")

# plot 1: data + fit
plt.figure()
plt.scatter(d, v, label="Data")
plt.plot(d, v_fit, color="red", label=f"Fit (H0 ≈ {H0_est:.2f} km/s/Mpc)")
plt.legend()
plt.xlabel("Distance (Mpc)")
plt.ylabel("Recession Velocity (km/s)")
plt.title("Galaxy Distance vs Recession Velocity")
plt.savefig("redshift_distance.png", dpi=300)
plt.show()

# plot 2: residuals
plt.figure()
plt.scatter(d, residuals)
plt.axhline(0)
plt.xlabel("Distance (Mpc)")
plt.ylabel("Residual (km/s)")
plt.title("Fit Residuals")
plt.savefig("residuals.png", dpi=300)
plt.show()
