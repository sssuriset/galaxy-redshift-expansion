# Galaxy Redshift Expansion

This project estimates the Hubble constant from a small redshift and distance dataset. The current pipeline uses a synthetic galaxy sample, which makes it easier to test how regression choices, noise, and outliers affect the fitted value of H0.

This is not meant to be a precision cosmology measurement. It is a small numerical analysis project built around the Hubble law.

## What the code does

The main script converts redshift to recession velocity using the low redshift approximation:

    v = cz

It then compares two fits:

1. A weighted fit through the origin
2. An unweighted linear fit with an intercept

The script also runs a bootstrap estimate for H0, removes high residual points as a sensitivity test, and saves plots and CSV files.

The advanced analysis script adds:

1. A classical versus relativistic velocity comparison
2. Monte Carlo sampling with distance and redshift uncertainty
3. Jackknife influence testing
4. Reduced chi square for the baseline weighted fit

## Dataset note

The main dataset is synthetic. If `data/galaxy_redshift_data.csv` is missing, the code creates it.

The synthetic data include distance uncertainty, redshift uncertainty, peculiar velocity noise, and one injected outlier. That makes the project useful for testing the fitting process, but the result should not be treated as a measured value of H0.

The folder `redshift_real_data_scratch/` contains early attempts at using larger real catalog files. Those files are kept for future work, but they are not part of the current finished pipeline.

## How to run

Install the dependencies:

    python3 -m pip install -r requirements.txt

Run the main analysis:

    python3 src/main.py

Run the additional diagnostics:

    python3 src/advanced_analysis.py

## Outputs

The scripts save figures and tables in `outputs/`.

Main outputs:

- `results.csv`
- `model_comparison.csv`
- `removed_outliers.csv`
- `hubble_fit_comparison.png`
- `residuals.png`

Advanced outputs:

- `advanced_results.csv`
- `monte_carlo_h0_distribution.png`
- `classical_vs_relativistic_velocity.png`
- `jackknife_h0_influence.png`
- `jackknife_influence.csv`

## Results

The synthetic sample is generated with an input H0 near 67.5 km/s/Mpc. The recovered value changes depending on the fitting method and on whether the injected outlier is included.

The weighted fit is the baseline result because it uses the redshift derived velocity uncertainty. Distance uncertainty is shown in the plots and used in the Monte Carlo diagnostic, but it is not included in the baseline weighted regression.

The outlier removal step is only a sensitivity check. It is not a formal outlier detection method for survey data.

## Method

For nearby galaxies, the code estimates recession velocity with:

    v = cz

The fitted Hubble law model is:

    v = H0 d

where v is recession velocity, d is distance, and H0 is the fitted slope.

The bootstrap test resamples the galaxy list and refits H0 many times. This gives a rough estimate of how much the result depends on the finite sample. The jackknife test removes one galaxy at a time to show which points have the most influence on the fitted slope.

## Limitations

This project uses a small synthetic dataset. It does not include survey selection effects, calibration uncertainty, correlated distance errors, redshift frame corrections, or a full cosmological distance relation.

The real catalog files in the scratch folder still need cleaning before they can be used in the main pipeline.

## Future work

- Clean the real catalog files into a usable distance redshift table
- Add a cited observational dataset
- Replace the simple outlier cutoff with a more defensible robust fitting method
- Compare low redshift and relativistic velocity models across a wider redshift range
- Combine velocity and distance uncertainty in the main fit
