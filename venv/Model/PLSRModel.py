try:
    from osgeo import gdal, ogr, gdal_array # I/O image data
    import numpy as np # math and array handling
    import matplotlib.pyplot as plt # plot figures
    import pandas as pd # handling large data as table sheets
    from joblib import dump, load
    from operator import itemgetter
    import joblib
    import sys, logging, traceback
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict
    sys.path.append(r"F:\Work\Maptor\venv\Model")
    from ReportModule import ReportModule

except Exception as e:
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)


class PLSR():
    rt = ReportModule()


    def PLSR_Regressor(self,features,X,y):
        try:
            # print("Inside Regessor")

            mse = []
            component = np.arange(1, features.shape[1])

            for i in component:
                pls = PLSRegression(n_components=i)
                # Cross-validation
                y_cv = cross_val_predict(pls, X, y, cv=10)
                mse.append(mean_squared_error(y, y_cv))
                comp = 100*(i+1)/40

            # Calculate and print the position of minimum in MSE
            msemin = np.argmin(mse)
            # Define PLS object with optimal number of components
            try:
                pls_opt = PLSRegression(n_components = msemin + 1)

                pls_opt.fit(X, y)
                y_c = pls_opt.predict(X)
                y_cv = cross_val_predict(pls_opt, X, y, cv=X.shape[0])
            except Exception as e:
                print(e)
                sys.exit(0)
            # Fir to the entire dataset
            score_c = r2_score(y, y_c)
            score_cv = r2_score(y, y_cv)

            # Calculate mean squared error for calibration and cross validation
            mse_c = mean_squared_error(y, y_c)
            mse_cv = mean_squared_error(y, y_cv)

            return [pls_opt,mse,msemin,component,y,y_c,y_cv,score_c,score_cv]

        except Exception as e:
            print(e)
            logging.debug("Exception occurred", exc_info=True)
            logging.error(traceback.print_exc())
            logging.info((traceback.print_exc()))
            sys.exit(0)

    def PLSR_vip(self,model):
        try:
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
        except ValueError as e:
            print(e)
            logging.error("Exception occurred", exc_info=True)
            sys.exit(0)

    def PLSR_Predict(self,doc, score_cv, importance, img, y_c, y, y_cv,labels, pls_opt):
        try:
            # imp = {}
            # for i in range(features.shape[1]):
            #     print('Band {}: {}'.format(i + 1, importance[i]))
            #     imp['Band{}'.format(i + 1)] = importance[i]

            # In[14]:

            # sorted_imp = dict(sorted(imp.items(), key=itemgetter(1), reverse=True))
            # sorted_imp

            # In[75]:

            # Plot regression and figures of merit

            rangey = max(y) - min(y)
            rangex = max(y_c) - min(y_c)

            # Fit a line to the CV vs response
            # z = np.polyfit(y, y_c, 1)
            # with plt.style.context(('ggplot')):
            #     fig, ax = plt.subplots(figsize=(9, 5))
            #     ax.scatter(y_c, y, c='red', edgecolors='k')
            #     # Plot the best fit line
            #     ax.plot(np.polyval(z, y), y, c='blue', linewidth=1)
            #     # Plot the ideal 1:1 line
            #     ax.plot(y, y, color='green', linewidth=1)
            #     plt.title('$R^{2}$ (CV): ' + str(score_cv))
            #     plt.xlabel('Predicted')
            #     plt.ylabel('Measured')
            #     plt.legend(['best fit line', 'ideal 1:1 line', 'samples'])
            #
            #     plt.show()
            #
            # # In[16]:
            #
            # # Calculate the absolute errors
            errors = abs(y - y_cv)
            # # Print out the mean absolute error (mae)
            #
            # # Print out the mean absolute error (mae)
            #
            # print('-------------')
            # print('n of the test data: {}'.format(len(labels)))
            # print('Mean of the variable: {:.2f}'.format(np.mean(labels)))
            # print('Standard deviation of the variable: {:.2f}'.format(np.std(labels)))
            # print('-------------')
            # print('Mean Absolute Error: {:.2f}'.format(np.mean(errors)))

            mse = mean_squared_error(y, y_cv)
          #  print("1 HERE!!!!!! mse" + str(mse))
            # print('Mean squared error: {:.2f}'.format(mse))
            # print('RMSE: {:.2f}'.format(np.sqrt(mse)))
            # print('RPD: {:.2f} | How often does RMSE of Prediction fit in the Standard Deviation of the samples'.format(
            #     np.std(labels) / np.sqrt(mse)))
            '''
            To put our predictions in perspective, we can calculate an accuracy using
            the mean average percentage error subtracted from 100 %.
            '''

            # Calculate mean absolute percentage error (MAPE)
            mape = 100 * (errors / labels)
            # Calculate and display accuracy
            accuracy = 100 - np.mean(mape)
            print('mean absolute percentage error (MAPE): {:.2f} %'.format(np.mean(mape)))
            print('accuracy (100 % - mape): {:.2f} %'.format(accuracy))
            print('-------------')
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination r²: {:.2f}'.format(r2_score(y, y_cv)))

            # In[17]:

            # Predicting the rest of the image

            # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
            new_shape = (img.shape[0] * img.shape[1], img.shape[2])
            img_as_array = img[:, :, :].reshape(new_shape)

            print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

            img_as_array = np.nan_to_num(img_as_array)

            # In[18]:

            prediction_ = pls_opt.predict(img_as_array)

            # In[19]:

            prediction = prediction_.reshape(img[:, :, 0].shape)
            print('Reshaped back to {}'.format(prediction.shape))

            # In[20]:

            # generate mask image from red band
            # mask = np.copy(img[:, :, 0])
            # mask[mask > 0.0] = 1.0  # all actual pixels have a value of 1.0
            #
            # # plot mask
            #
            # plt.imshow(mask)
            #
            # # In[21]:
            #
            # # mask classification an plot
            #
            # prediction_ = prediction * mask
            #
            # plt.subplot(121)
            # plt.imshow(prediction, cmap=plt.cm.Spectral)
            # plt.title('classification unmasked')
            #
            # plt.subplot(122)
            # plt.imshow(prediction_, cmap=plt.cm.Spectral)
            # plt.title('classification masked')
            #
            # plt.show()
            #
            # plt.subplot(121)
            # plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            # plt.title('RS image - first band')
            #
            # plt.subplot(122)
            # plt.imshow(prediction_, cmap=plt.cm.Spectral, vmin=y.min(), vmax=y.max())
            # plt.title('Prediction')
            # plt.colorbar()
            #
            # plt.show()

            # In[22]:

            # Save regression

            cols = img.shape[1]
            rows = img.shape[0]

            return [cols,rows,doc,prediction,prediction_]

        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)
            sys.exit(0)

    def PLSR_SaveImage(self,prediction_map,img,img_ds,prediction_):
        try:
            cols = img.shape[1]
            rows = img.shape[0]

            driver = gdal.GetDriverByName("gtiff")
            outdata = driver.Create(prediction_map, cols, rows, 1, gdal.GDT_Float32)
            outdata.SetGeoTransform(img_ds.GetGeoTransform())  ##sets same geotransform as input
            outdata.SetProjection(img_ds.GetProjection())  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(prediction_)
            outdata.FlushCache()  ##saves to disk!!
            print('Image saved to: {}'.format(prediction_map))
        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)
            sys.exit(0)

    def PLSR_SaveModel(self, model, file_name):
        try:
            joblib.dump(model, file_name)
        except ValueError as e:
            print(e)
            logging.error("Exception occurred", exc_info=True)
            print("Could not save image")
            sys.exit(0)
