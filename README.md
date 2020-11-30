# Maptor - Machine learning Regression and Classification for Remote Sensing Data
Classification and regression on remotely sensed data - ***it is as simple as that!***

- ***You only need (1) a remote sensing dataset as tiff and (2) a ground truth dataset as shape***
- ***polygon shape file for classification***
- ***point shape file for regression***


# NEW RELEASE!!! Maptor 1.4beta
Finally, we are pleased to inform you, that our brand new software [Maptor](https://datenportal.wetscapes.de/dataset/maptor-0-0) has now been released as a beta version (2020-11-11).

***Main features: Random Forest Classification and Regression as well as Partial Least Squares Regression on remote sensing data, for both large and sparse sampling sizes***

![Maptor](http://flobeyer.de/img/Maptor_Screenshot.JPG "Maptor 1.4beta")

[Please download and test Maptor 1.4beta here!](https://datenportal.wetscapes.de/dataset/maptor-0-0)

# (new) features
--> NOT necessary anymore!!! -> defining the environment variable

## Classification

- ***Random Forest classfication***
- ground truth data as shape.shp (classes as integer)
- set your number of trees
- save trained model(.sav)
- save report.pdf and classification.tif

## Regression

### large sampling sizes
- ***Random Forest (RF) Regressor***
- ground truth data as shape.shp (samples as integer or float)
- split your samples 0.25 -> 75 % Training samples / 25 % Validation samples
- save trained model(.sav)
- save report.pdf and regression.tif

### sparse sampling sizes
- ***Partial Least Squares Regression (PLSR)***
- ground truth data as shape.shp (samples as integer or float)
- LOOCV -> Leave-One-Out Crossvalidation
- automatic selection of best number of latent components
- save trained model(.sav)
- save report.pdf and regression.tif

## known bugs
- memory error with very big image files (depending on RAM size) 
***found something/suggestions? please contact florian.beyer@uni-rostock.de***


# upcoming features

- linux and macOS
- PLSR for large sample sizes (regression)
- RF für sparse sampling sizes (regression)
- load trained model
- online user guide
- automatic parameter tuning for Random Forest Classification and Regression (grid search)
