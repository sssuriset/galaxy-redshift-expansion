import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C_KM_S = 299792.458


def load():
    path = "data/galaxy_redshift_data.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing data/galaxy_redshift_data.csv. Run python3 src/main.py first."
        )

    data = pd.read_csv(path)

    data["velocity_classical_km_s"] = C_KM_S * data["redshift"]

    z = data["redshift"]
    data["velocity_relativistic_km_s"] = C_KM_S * (((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1))

    data["velocity_uncertainty_km_s"] = C_KM_S * data["redshift_uncertainty"]

    return data


def fit_weighted(distance, velocity, velocity_uncertainty):
    weights = 1 / velocity_uncertainty**2
    return np.sum(weights * distance * velocity) / np.sum(weights * distance**2)


def chi2_reduced(distance, velocity, velocity_uncertainty, h0):
    model = h0 * distance
    residuals = velocity - model
    chi_square = np.sum((residuals / velocity_uncertainty) ** 2)
    degrees_of_freedom = len(distance) - 1

    return chi_square / degrees_of_freedom


def monte_carlo(data, trials=10000):
    np.random.seed(31)

    h0_values = []

    distance = data["distance_mpc"].to_numpy()
    distance_uncertainty = data["distance_uncertainty_mpc"].to_numpy()
    redshift = data["redshift"].to_numpy()
    redshift_uncertainty = data["redshift_uncertainty"].to_numpy()
    velocity_uncertainty = data["velocity_uncertainty_km_s"].to_numpy()

    for _ in range(trials):
        sampled_distance = np.random.normal(distance, distance_uncertainty)
        sampled_redshift = np.random.normal(redshift, redshift_uncertainty)
        sampled_velocity = C_KM_S * sampled_redshift

        valid = sampled_distance > 0

        h0 = fit_weighted(
            sampled_distance[valid],
            sampled_velocity[valid],
            velocity_uncertainty[valid]
        )

        h0_values.append(h0)

    return np.array(h0_values)


def jackknife(data):
    distance = data["distance_mpc"].to_numpy()
    velocity = data["velocity_classical_km_s"].to_numpy()
    velocity_uncertainty = data["velocity_uncertainty_km_s"].to_numpy()

    full_h0 = fit_weighted(distance, velocity, velocity_uncertainty)

    rows = []

    for i in range(len(data)):
        mask = np.ones(len(data), dtype=bool)
        mask[i] = False

        h0_without_one = fit_weighted(
            distance[mask],
            velocity[mask],
            velocity_uncertainty[mask]
        )

        rows.append({
            "removed_galaxy_id": data.iloc[i]["galaxy_id"],
            "h0_without_galaxy": h0_without_one,
            "delta_h0_from_full": h0_without_one - full_h0,
            "absolute_delta_h0": abs(h0_without_one - full_h0)
        })

    return pd.DataFrame(rows).sort_values("absolute_delta_h0", ascending=False)


def plot_monte_carlo(h0_values):
    plt.figure(figsize=(10, 5))
    plt.hist(h0_values, bins=40)
    plt.xlabel("H0 (km/s/Mpc)")
    plt.ylabel("Monte Carlo count")
    plt.title("Monte Carlo H0 Uncertainty Distribution")
    plt.tight_layout()
    plt.savefig("outputs/monte_carlo_h0_distribution.png", dpi=300)
    plt.close()


def plot_velocity_check(data):
    plt.figure(figsize=(10, 5))
    plt.scatter(
        data["velocity_classical_km_s"],
        data["velocity_relativistic_km_s"]
    )

    limit = max(data["velocity_classical_km_s"].max(), data["velocity_relativistic_km_s"].max())
    plt.plot([0, limit], [0, limit], linestyle="--")

    plt.xlabel("Classical velocity cz (km/s)")
    plt.ylabel("Relativistic velocity estimate (km/s)")
    plt.title("Classical and Relativistic Velocity Comparison")
    plt.tight_layout()
    plt.savefig("outputs/classical_vs_relativistic_velocity.png", dpi=300)
    plt.close()


def plot_jackknife(jackknife_table):
    plt.figure(figsize=(10, 5))
    plt.bar(jackknife_table["removed_galaxy_id"], jackknife_table["delta_h0_from_full"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Removed galaxy")
    plt.ylabel("Change in H0 (km/s/Mpc)")
    plt.title("Jackknife Influence on H0")
    plt.tight_layout()
    plt.savefig("outputs/jackknife_h0_influence.png", dpi=300)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    data = load()

    distance = data["distance_mpc"].to_numpy()
    velocity_classical = data["velocity_classical_km_s"].to_numpy()
    velocity_relativistic = data["velocity_relativistic_km_s"].to_numpy()
    velocity_uncertainty = data["velocity_uncertainty_km_s"].to_numpy()

    h0_classical = fit_weighted(distance, velocity_classical, velocity_uncertainty)
    h0_relativistic = fit_weighted(distance, velocity_relativistic, velocity_uncertainty)

    reduced_chi2_classical = chi2_reduced(
        distance,
        velocity_classical,
        velocity_uncertainty,
        h0_classical
    )

    h0_mc = monte_carlo(data)
    mc_lower, mc_median, mc_upper = np.percentile(h0_mc, [2.5, 50, 97.5])

    jackknife_table = jackknife(data)

    advanced_results = pd.DataFrame({
        "metric": [
            "fit_weighted_classical_velocity",
            "fit_weighted_relativistic_velocity",
            "difference_relativistic_minus_classical",
            "chi2_reduced_classical_fit",
            "monte_carlo_lower_95",
            "monte_carlo_median",
            "monte_carlo_upper_95",
            "largest_single_galaxy_h0_shift"
        ],
        "value": [
            h0_classical,
            h0_relativistic,
            h0_relativistic - h0_classical,
            reduced_chi2_classical,
            mc_lower,
            mc_median,
            mc_upper,
            jackknife_table.iloc[0]["absolute_delta_h0"]
        ]
    })

    advanced_results.to_csv("outputs/advanced_results.csv", index=False)
    jackknife_table.to_csv("outputs/jackknife_influence.csv", index=False)

    plot_monte_carlo(h0_mc)
    plot_velocity_check(data)
    plot_jackknife(jackknife_table)

    print("Classical weighted H0:", round(h0_classical, 3), "km/s/Mpc")
    print("Relativistic weighted H0:", round(h0_relativistic, 3), "km/s/Mpc")
    print("Reduced chi-square:", round(reduced_chi2_classical, 3))
    print("Monte Carlo 95% interval:", round(mc_lower, 3), "to", round(mc_upper, 3))
    print("Most influential galaxy:", jackknife_table.iloc[0]["removed_galaxy_id"])
    print("Largest H0 shift:", round(jackknife_table.iloc[0]["absolute_delta_h0"], 3), "km/s/Mpc")
    print("\nSaved outputs/advanced_results.csv")
    print("Saved outputs/jackknife_influence.csv")


if __name__ == "__main__":
    main()
