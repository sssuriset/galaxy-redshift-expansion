import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

C_KM_S = 299792.458


def make_sample_data(path):
    np.random.seed(12)

    distance_mpc = np.array([8, 12, 18, 25, 32, 40, 55, 70, 85, 100, 120, 145, 170, 200])
    true_h0 = 67.5

    peculiar_velocity = np.random.normal(0, 180, size=len(distance_mpc))
    velocity_km_s = true_h0 * distance_mpc + peculiar_velocity
    redshift = velocity_km_s / C_KM_S

    distance_uncertainty_mpc = 0.08 * distance_mpc + 1.5
    redshift_uncertainty = np.full(len(distance_mpc), 0.00035)

    data = pd.DataFrame({
        "galaxy_id": [f"G{i+1:02d}" for i in range(len(distance_mpc))],
        "distance_mpc": distance_mpc,
        "distance_uncertainty_mpc": distance_uncertainty_mpc,
        "redshift": redshift,
        "redshift_uncertainty": redshift_uncertainty
    })

    outlier = pd.DataFrame({
        "galaxy_id": ["G15_outlier"],
        "distance_mpc": [95],
        "distance_uncertainty_mpc": [9],
        "redshift": [(67.5 * 95 + 1200) / C_KM_S],
        "redshift_uncertainty": [0.00035]
    })

    data = pd.concat([data, outlier], ignore_index=True)
    data.to_csv(path, index=False)


def load_data():
    os.makedirs("data", exist_ok=True)
    path = "data/galaxy_redshift_data.csv"

    if not os.path.exists(path):
        make_sample_data(path)

    data = pd.read_csv(path)
    data["velocity_km_s"] = C_KM_S * data["redshift"]
    data["velocity_uncertainty_km_s"] = C_KM_S * data["redshift_uncertainty"]

    return data


def weighted_h0_fit(distance, velocity, velocity_uncertainty):
    weights = 1 / velocity_uncertainty**2
    h0 = np.sum(weights * distance * velocity) / np.sum(weights * distance**2)
    intercept = 0

    model = h0 * distance + intercept
    residuals = velocity - model

    return h0, intercept, residuals


def unweighted_fit(distance, velocity):
    result = linregress(distance, velocity)
    model = result.slope * distance + result.intercept
    residuals = velocity - model

    return result.slope, result.intercept, residuals


def bootstrap_h0(distance, velocity, velocity_uncertainty, samples=5000):
    np.random.seed(22)
    h0_values = []

    n = len(distance)

    for _ in range(samples):
        index = np.random.randint(0, n, n)
        h0, _, _ = weighted_h0_fit(
            distance[index],
            velocity[index],
            velocity_uncertainty[index]
        )
        h0_values.append(h0)

    lower, upper = np.percentile(h0_values, [2.5, 97.5])
    return lower, upper, np.array(h0_values)


def residual_stats(residuals):
    return {
        "mean_residual_km_s": np.mean(residuals),
        "std_residual_km_s": np.std(residuals, ddof=1),
        "rms_residual_km_s": np.sqrt(np.mean(residuals**2)),
        "max_abs_residual_km_s": np.max(np.abs(residuals))
    }


def remove_outliers(data, residuals, sigma_cut=2.0):
    std = np.std(residuals, ddof=1)
    mask = np.abs(residuals) < sigma_cut * std
    return data[mask].copy(), data[~mask].copy()


def save_hubble_plot(data, h0_weighted, intercept_unweighted, h0_unweighted, clean_data=None, h0_clean=None):
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        data["distance_mpc"],
        data["velocity_km_s"],
        xerr=data["distance_uncertainty_mpc"],
        yerr=data["velocity_uncertainty_km_s"],
        fmt="o",
        capsize=3,
        label="Galaxy data"
    )

    x = np.linspace(0, data["distance_mpc"].max() * 1.08, 300)

    plt.plot(x, h0_weighted * x, label=f"Weighted fit through origin: H0 = {h0_weighted:.2f}")
    plt.plot(x, h0_unweighted * x + intercept_unweighted, linestyle="--", label=f"Unweighted fit: H0 = {h0_unweighted:.2f}")

    if clean_data is not None and h0_clean is not None:
        plt.plot(x, h0_clean * x, linestyle=":", label=f"Weighted fit without outlier: H0 = {h0_clean:.2f}")

    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Recession velocity (km/s)")
    plt.title("Galaxy Redshift-Distance Relation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/hubble_fit_comparison.png", dpi=300)
    plt.close()


def save_residual_plot(data, residuals, clean_residuals=None, clean_data=None):
    plt.figure(figsize=(10, 5))

    plt.axhline(0, linestyle="--")
    plt.scatter(data["distance_mpc"], residuals, label="Weighted-fit residuals")

    if clean_data is not None and clean_residuals is not None:
        plt.scatter(clean_data["distance_mpc"], clean_residuals, marker="x", label="Residuals after outlier removal")

    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Residual velocity (km/s)")
    plt.title("Residuals from Hubble-Law Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/residuals.png", dpi=300)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    data = load_data()

    distance = data["distance_mpc"].to_numpy()
    velocity = data["velocity_km_s"].to_numpy()
    velocity_uncertainty = data["velocity_uncertainty_km_s"].to_numpy()

    h0_weighted, intercept_weighted, residuals_weighted = weighted_h0_fit(
        distance,
        velocity,
        velocity_uncertainty
    )

    h0_unweighted, intercept_unweighted, residuals_unweighted = unweighted_fit(
        distance,
        velocity
    )

    clean_data, removed_data = remove_outliers(data, residuals_weighted)

    clean_distance = clean_data["distance_mpc"].to_numpy()
    clean_velocity = clean_data["velocity_km_s"].to_numpy()
    clean_velocity_uncertainty = clean_data["velocity_uncertainty_km_s"].to_numpy()

    h0_clean, _, residuals_clean = weighted_h0_fit(
        clean_distance,
        clean_velocity,
        clean_velocity_uncertainty
    )

    ci_lower, ci_upper, _ = bootstrap_h0(
        distance,
        velocity,
        velocity_uncertainty
    )

    stats_weighted = residual_stats(residuals_weighted)
    stats_clean = residual_stats(residuals_clean)

    results = pd.DataFrame({
        "metric": [
            "weighted_h0_km_s_mpc",
            "unweighted_h0_km_s_mpc",
            "weighted_h0_without_outlier_km_s_mpc",
            "weighted_h0_95_ci_lower",
            "weighted_h0_95_ci_upper",
            "weighted_mean_residual_km_s",
            "weighted_rms_residual_km_s",
            "weighted_max_abs_residual_km_s",
            "clean_mean_residual_km_s",
            "clean_rms_residual_km_s",
            "removed_outlier_count"
        ],
        "value": [
            h0_weighted,
            h0_unweighted,
            h0_clean,
            ci_lower,
            ci_upper,
            stats_weighted["mean_residual_km_s"],
            stats_weighted["rms_residual_km_s"],
            stats_weighted["max_abs_residual_km_s"],
            stats_clean["mean_residual_km_s"],
            stats_clean["rms_residual_km_s"],
            len(removed_data)
        ]
    })

    results.to_csv("outputs/results.csv", index=False)

    model_comparison = pd.DataFrame({
        "model": [
            "weighted_origin_fit",
            "unweighted_intercept_fit",
            "weighted_origin_fit_without_outlier"
        ],
        "h0_km_s_mpc": [
            h0_weighted,
            h0_unweighted,
            h0_clean
        ],
        "intercept_km_s": [
            intercept_weighted,
            intercept_unweighted,
            0
        ],
        "rms_residual_km_s": [
            stats_weighted["rms_residual_km_s"],
            residual_stats(residuals_unweighted)["rms_residual_km_s"],
            stats_clean["rms_residual_km_s"]
        ]
    })

    model_comparison.to_csv("outputs/model_comparison.csv", index=False)
    removed_data.to_csv("outputs/removed_outliers.csv", index=False)

    save_hubble_plot(
        data,
        h0_weighted,
        intercept_unweighted,
        h0_unweighted,
        clean_data,
        h0_clean
    )

    save_residual_plot(
        data,
        residuals_weighted,
        residuals_clean,
        clean_data
    )

    print("Weighted H0:", round(h0_weighted, 2), "km/s/Mpc")
    print("Unweighted H0:", round(h0_unweighted, 2), "km/s/Mpc")
    print("Weighted H0 without outlier:", round(h0_clean, 2), "km/s/Mpc")
    print("95% bootstrap CI:", round(ci_lower, 2), "to", round(ci_upper, 2), "km/s/Mpc")
    print("Removed outliers:", len(removed_data))
    print("\nSaved outputs/results.csv")
    print("Saved outputs/model_comparison.csv")


if __name__ == "__main__":
    main()
