try:
    import sys,os
    # sys.path.append(r"F:\Work\Maptor\venv\Model")
    from ReportModule import ReportModule
    # sys.path.append(r"F:\Work\Maptor\venv\Model")
    from InputController import InputController
    # sys.path.append(r"F:\Work\Maptor\venv\HelpingModel")
    from RFHelper import RFHelper
    # sys.path.append(r"..\HelpingModel")
    from RegRptHelper import RegressionReportHelper
    from RF_NSS_Controller import RF_NSS_Controller
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QFileDialog,QMessageBox
    from osgeo import ogr

    from osgeo import gdal, ogr, gdal_array  # I/O image data
    import numpy as np  # math and array handling
    import matplotlib.pyplot as plt  # plot figures
    import pandas as pd  # handling large data as table sheets
    import joblib
    from joblib import dump, load
    from operator import itemgetter

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict

    from sklearn import preprocessing
    import seaborn as sns

except Exception as e:
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)


class RFR_SSS_Model():

    def RFR_Regressor(self,X,y,est,features,labels):
        try:
            randomState = 35
            RFR = RandomForestRegressor(bootstrap=True,
                                    criterion='mse',
                                    max_depth=None,
                                    max_features='auto',
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0.0,
                                    min_impurity_split=None,
                                    min_samples_leaf=1,
                                    min_samples_split=2,
                                    min_weight_fraction_leaf=0.0,
                                    n_estimators=est,
                                    n_jobs=-1,  # using all cores
                                    oob_score=True,
                                    random_state=randomState,
                                    verbose=1,
                                    warm_start=False)
            RFR.fit(features, labels);
            y_c = RFR.predict(X)

            y_cv = cross_val_predict(RFR, X, y, cv=X.shape[0],n_jobs=-1,verbose=2)

            return[y_c,y_cv,RFR]

        except ValueError as e:
            print(e)

    def RFR_prediction(self,RFR,img_as_array):
        try:
            return RFR.predict(img_as_array)
        except ValueError as e:
            print(e)


    def save_model(self, model, file_name):
        try:
            joblib.dump(model, file_name)
        except ValueError as e:
            print(e)
            print("Could not save image")

    def RF_save_PredictionImage(self,prediction_,prediction_map, img, img_ds):
        # Save regression
        try:
            cols = img.shape[1]
            rows = img.shape[0]

            prediction_.astype(np.float32)

            driver = gdal.GetDriverByName("gtiff")
            outdata = driver.Create(prediction_map, cols, rows, 1, gdal.GDT_Float32)
            outdata.SetGeoTransform(img_ds.GetGeoTransform())  ##sets same geotransform as input
            outdata.SetProjection(img_ds.GetProjection())  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(prediction_)
            outdata.FlushCache()  ##saves to disk!!
            print('Image saved to: {}'.format(prediction_map))
        except ValueError as e:
            print(e)