# Galaxy Redshift Expansion

A Hubble constant estimation study on a controlled synthetic galaxy sample. Because the data generator plants the truth (H0 = 67.5 km/s/Mpc, with distance and redshift uncertainty, peculiar velocity scatter, and one injected outlier), every fitting choice can be judged against a known answer instead of against hope.

The headline comparison: a weighted fit through the origin on the full sample returns 68.39, removing the planted outlier with a 2-sigma residual cut moves it to 67.60, and the injected truth is 67.5. One bad galaxy at 95 Mpc shifts the answer by 0.8 km/s/Mpc, which is the point of the exercise.

## Run

```bash
python3 -m pip install numpy pandas matplotlib scipy
python3 src/main.py                 # fits, bootstrap, outlier cut
python3 src/advanced_analysis.py    # relativistic comparison, Monte Carlo, jackknife
```

`data/galaxy_redshift_data.csv` ships with the repo; if deleted, `src/main.py` regenerates it from a fixed seed, so runs reproduce exactly. Figures and CSV tables land in `outputs/`.

## What gets measured

`main.py` converts redshift to velocity with v = cz, fits H0 three ways (weighted through the origin, unweighted with a free intercept, and weighted after the residual cut), bootstraps a 95 percent confidence interval (66.76 to 71.19), and records residual statistics before and after the cut (RMS drops from 359 to 208 km/s when the outlier goes).

`advanced_analysis.py` swaps in the relativistic velocity conversion, which lowers the fit by 1.13 km/s/Mpc even at these low redshifts, runs a Monte Carlo over both measurement uncertainties (95 percent interval 63.3 to 73.1), computes jackknife influence per galaxy, and reports the reduced chi-square of the baseline fit (12.5, honestly poor, driven by peculiar velocities the error model ignores).

## Reading the numbers

The free intercept lands at 3.7 km/s against a scatter of hundreds, so the data do not demand an offset, consistent with the Hubble law passing through the origin. The reduced chi-square far above 1 shows the distance errors alone cannot explain the scatter; peculiar velocity acts as an unmodeled error term, which is exactly what it does in real low-redshift samples. Since the sample is synthetic and small, the value of this project is the sensitivity analysis, not the H0 value itself.
