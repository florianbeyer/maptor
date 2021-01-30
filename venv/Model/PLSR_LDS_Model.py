
try:
    from osgeo import gdal, ogr, gdal_array  # I/O image data
    import numpy as np  # math and array handling
    import matplotlib.pyplot as plt  # plot figures
    import pandas as pd  # handling large data as table sheets
    from joblib import dump, load
    from operator import itemgetter
    import sys, os
    from sklearn.model_selection import train_test_split
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict
    sys.path.append(r"F:\Work\Maptor\venv\Model")
    from ReportModule import ReportModule
    from sklearn import preprocessing
    import joblib
except Exception as e:
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)
class PLSR_LDS_Model():

    def ComponentRegressor(self,features,X,y):
        mse = []
        component = np.arange(1, features.shape[1])
        for i in component:
            pls = PLSRegression(n_components=i)
            y_cv = cross_val_predict(pls, X, y, cv=10,n_jobs=-1,verbose=2)
            mse.append(mean_squared_error(y, y_cv))
        return [mse,component]

    def Regressor(self, suggested_comp, train_features, train_labels):

        plsr = PLSRegression(n_components=suggested_comp,
                             scale=True,
                             max_iter=500,
                             tol=1e-06,
                             copy=True)
        plsr.fit(train_features, train_labels)
        print(plsr.score(train_features, train_labels))
        return plsr

    def bandimportance(self,train_features,plsr):
        importance = self.vip(plsr)
        return importance


    def testpredict(self, plsr,test_features):
        predictions_test_ds = plsr.predict(test_features)
        return predictions_test_ds

    def finaltraining(self,plsr,features,labels):
        plsr.fit(features, labels)
        return plsr

    def finalprediction(self,plsr,img_as_array):
        prediction_ = plsr.predict(img_as_array)
        return prediction_

    def vip(self,model):
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    def saveimage(self,img,prediction_,prediction_map,img_ds):
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

    def savemodel(self, model, file_name):
        try:
            joblib.dump(model, file_name)
        except ValueError as e:
            print(e)
            logging.error("Exception occurred", exc_info=True)
            print("Could not save image")
            sys.exit(0)