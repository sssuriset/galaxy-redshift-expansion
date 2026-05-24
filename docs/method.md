# Method

This project estimates the Hubble constant using galaxy redshift and distance data.

## Redshift to Velocity

For small redshift values, recession velocity is approximated as:

    v = cz

where v is recession velocity, c is the speed of light, and z is redshift. This approximation is valid only at low redshift. At larger redshift, a full cosmological distance-redshift relation would be required.

## Linear Model

The basic Hubble-law model is:

    v = H0 d

where H0 is the Hubble constant and d is distance in megaparsecs. Since the model predicts zero recession velocity at zero distance, the weighted fit is performed through the origin.

## Weighted Regression

The weighted fit uses velocity uncertainty to give less influence to measurements with larger uncertainty. The fitted slope gives the estimated value of H0 in km/s/Mpc.

## Bootstrap Confidence Interval

The script estimates a 95 percent confidence interval for H0 using bootstrap resampling. It repeatedly resamples the galaxy data with replacement, refits H0, and takes the 2.5th and 97.5th percentiles of the fitted values.

## Outlier Comparison

The script identifies outliers using residual size relative to the residual standard deviation. It compares the H0 estimate before and after outlier removal.

## Peculiar Velocities

Galaxy velocities are not caused only by cosmic expansion. Galaxies also have local motions caused by gravitational interactions with nearby structures. These peculiar velocities add scatter to the redshift-distance relation and can bias H0 estimates, especially for nearby galaxies where the expansion velocity is small.
