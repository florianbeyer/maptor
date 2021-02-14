from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from operator import itemgetter
import pandas as pd
import numpy as np
from osgeo import gdal
import joblib
import sys
# sys.path.append(r"..\HelpingModel")
from RegRptHelper import RegressionReportHelper

class RF_NSS_Model():
    helper = RegressionReportHelper()

    def RF_regressor(self,roi,img,attributes,test_size,est):
        try:
            ##test_size = 0.25 ## TEST
            randomState = 35 #35
            ##est = 100    ##1000
            n_samples = (roi > 0).sum()
            print(
                'We have {n} training samples'.format(n=n_samples))  # Subset the image dataset with the training image = X

            # Mask the classes on the training dataset = y
            # These will have n_samples rows
            X = img[roi > 0, :]
            y = roi[roi > 0]
            features = pd.DataFrame(X)
            band_names = []
            for i in range(X.shape[1]):
                # for i in range(0,2500):
                nband = "Band_" + str(i + 1)
                band_names.append(nband)

            features.columns = band_names
            print('The shape of our features is:', features.shape)
            print('The number of Spectra is:', features.shape[0])
            print('The number of bands is:', features.shape[1])

            features['value'] = y
            features.head()

            # Labels are the values we want to predict
            labels = np.array(features['value'])

            # Remove the labels from the features
            # axis 1 refers to the columns
            features = features.drop('value', axis=1)

            # Saving feature names for later use
            feature_list = list(features.columns)

            # Convert to numpy array
            features = np.array(features)

            # Split the data into training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                        test_size=test_size,
                                                                                        random_state=randomState)

            print('Training Features Shape:', train_features.shape)
            print('Training Labels Shape:', train_labels.shape)
            print('Testing Features Shape:', test_features.shape)
            print('Testing Labels Shape:', test_labels.shape)

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

            RFR.fit(train_features, train_labels)

            RFRScore = (RFR.score(train_features, train_labels))
            RFRScore = round(RFRScore,2)
            # band importances
            imp = {}
            for i in range(len(RFR.feature_importances_)):
                importance = round(RFR.feature_importances_[i]*100,2)
                print('Band {}: {}'.format(i+1, importance))
                imp['Band{}'.format(i+1)] = importance

            print(img.shape[0])
            print(img.shape[1])
            imageExtent = str(img.shape[0]) +" x "+ str(img.shape[1])
            self.helper.imageExtent = imageExtent
            self.helper.BandNumber = features.shape[1]
            self.helper.FieldName = attributes
            self.helper.TreesNo = est
            self.helper.SampleNo = n_samples
            self.helper.SplitSize = test_size
            self.helper.TrainingSample = train_labels.shape[0]
            self.helper.TestSample = test_labels.shape[0]
            self.helper.Coeff = str(RFRScore)
            self.helper.BandImportance = imp

            sorted_imp = dict(sorted(imp.items(), key=itemgetter(1), reverse=True))
            sorted_imp



            # Use the forest's predict method on the test data
            predictions_test_ds = RFR.predict(test_features)



            # Calculate the absolute errors
            errors = abs(predictions_test_ds - test_labels)
            # Print out the mean absolute error (mae)

            # Print out the mean absolute error (mae)


            print('-------------')
            print('n of the test data: {}'.format(len(test_labels)))
            self.helper.n_testdata= len(test_labels)
            print('Mean of the variable: {:.2f}'.format(np.mean(labels)))
            self.helper.meanvariable = np.mean(labels)



            #print('Standard deviation of the variable: {:.2f}'.format(np.std(labels)))
            self.helper.stdDev = np.std(labels)
            #print('-------------')
            #print('Mean Absolute Error: {:.2f}'.format(np.mean(errors)))
            self.helper.absError = np.mean(errors)
            mse = mean_squared_error(test_labels, predictions_test_ds)

            #print('Mean squared error: {:.2f}'.format(mse))
            self.helper.meanSqError = mse
            #print('RMSE: {:.2f}'.format(np.sqrt(mse)))
            self.helper.RMSE = np.sqrt(mse)

            '''
            To put our predictions in perspective, we can calculate an accuracy using
            the mean average percentage error subtracted from 100 %.
            '''
            # Calculate mean absolute percentage error (MAPE)
            mape = 100 * (errors / test_labels)
            # Calculate and display accuracy
            accuracy = 100 - np.mean(mape)
            #print('mean absolute percentage error (MAPE) / Accuracy: {:.2f}'.format(accuracy), '%.')
            self.helper.MAPE = accuracy
            #print('-------------')
            # The coefficient of determination: 1 is perfect prediction
           # print('Coefficient of determination r²: {:.2f}'.format(r2_score(test_labels, predictions_test_ds)))
            self.helper.coeffR = r2_score(test_labels, predictions_test_ds)


            print(type(test_labels))
            print(type(predictions_test_ds))

            self.helper.testlabels = test_labels
            self.helper.pred_test_ds = predictions_test_ds
            return [RFR,self.helper]

        except ValueError as e:
            print(e)
            print("Failed to apply regression")

        # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification

    def RF_prediction(self,RFR,img):
        try:
            new_shape = (img.shape[0] * img.shape[1], img.shape[2])
            img_as_array = img[:, :, :].reshape(new_shape)
            print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))
            img_as_array = np.nan_to_num(img_as_array)
            prediction_ = RFR.predict(img_as_array)
            prediction = prediction_.reshape(img[:, :, 0].shape)
            print('Reshaped back to {}'.format(prediction.shape))

            # Save regression
            cols = img.shape[1]
            rows = img.shape[0]

            prediction.astype(np.float32)
            return prediction
        except ValueError as e:
            print(e)
            print("Unable to apply Prediction on Regression")

    def RF_save_predictionImage(self,prediction,prediction_map,img,img_ds):
        try:
            cols = img.shape[1]
            rows = img.shape[0]
            driver = gdal.GetDriverByName("gtiff")
            outdata = driver.Create(prediction_map, cols, rows, 1, gdal.GDT_Float32)
            outdata.SetGeoTransform(img_ds.GetGeoTransform())  ##sets same geotransform as input
            outdata.SetProjection(img_ds.GetProjection())  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(prediction)
            outdata.FlushCache()  ##saves to disk!!
            print('Image saved to: {}'.format(prediction_map))
        except ValueError as e:
            print(e)
            print("Prediction Image Could not be saved..")

    def save_model(self, model, file_name):
        try:
            joblib.dump(model, file_name)
        except ValueError as e:
            print(e)
            print("Could not save image")



