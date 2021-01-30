from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report, accuracy_score # calculating measures for accuracy assessment
from osgeo import gdal
import joblib
import sys
sys.path.append(r"F:\Work\Maptor\venv\Model")
from ReportModule import ReportModule
import pandas as pd
import matplotlib.pyplot as plt


class ClassificationModel():
    Trees=500

    def set_trees(self,trees):
        self.Trees = trees

    def get_trees(self):
        return self.Trees

    def rf_classifier(self,img,img_ds,roi,trees):
        try:
            #rt = ReportModule()

            n_samples = (roi > 0).sum()
            #print('We have {n} training samples'.format(n=n_samples))

            # What are our classification labels?
            labels = np.unique(roi[roi > 0])
            #print('The training data include {n} classes: {classes}'.format(n=labels.size,
            #                                                                classes=labels))
            X = img[roi > 0, :]
            y = roi[roi > 0]

            # print('Our X matrix is sized: {sz}'.format(sz=X.shape))
            #
            # print('Our y array is sized: {sz}'.format(sz=y.shape))
             # Train Random Forest

            rf = RandomForestClassifier(n_estimators=trees, oob_score=True,verbose=2, n_jobs=-1 )
            X = np.nan_to_num(X)
            rf = rf.fit(X, y)

            ob_score = round(rf.oob_score_ * 100, 2)

            importance = {}
            bands = range(1, img_ds.RasterCount + 1)
            for b, imp in zip(bands, rf.feature_importances_):
                importance[b] = round(imp * 100, 2)
                #print('Band {b} importance: {imp} %'.format(b=b, imp=round(imp * 100, 2)))

            # Let's look at a crosstabulation to see the class confusion.
            # To do so, we will import the Pandas library for some help:
            # Setup a dataframe -- just like R
            # Exception Handling because of possible Memory Error
            try:
                df = pd.DataFrame()
                df['truth'] = y
                df['predict'] = rf.predict(X)

            except MemoryError:
                print('Crosstab not available ')
            else:
                # Cross-tabulate predictions
                table_M = pd.crosstab(df['truth'], df['predict'], margins=True)
                #print(table_M)
                del table_M['All']
                table_M = table_M.drop('All', axis=0)
                #doc = rt.prepare_section2(doc,img,roi,importance,table_M,trees,ob_score)

                return [rf,importance, table_M, ob_score]
        except ValueError as e:
            print(e)
            Print("Could not apply Classification...")

    def rf_prediction(self,img,rf):
        try:
            # # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
            new_shape = (img.shape[0] * img.shape[1], img.shape[2])
            img_as_array = img[:, :, :np.int(img.shape[2])].reshape(new_shape)

            #print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

            img_as_array = np.nan_to_num(img_as_array)

            # # In[9]:
            #
            # # Now predict for each pixel
            # # first prediction will be tried on the entire image
            # # if not enough RAM, the dataset will be sliced
            try:
                class_prediction = rf.predict(img_as_array)
            except MemoryError:
                slices = int(round(len(img_as_array) / 2))

                test = True

                while test == True:
                    try:
                        class_preds = list()

                        temp = rf.predict(img_as_array[0:slices + 1, :])
                        class_preds.append(temp)

                        for i in range(slices, len(img_as_array), slices):
                            print('{} %, derzeit: {}'.format((i * 100) / (len(img_as_array)), i))
                            temp = rf.predict(img_as_array[i + 1:i + (slices + 1), :])
                            class_preds.append(temp)

                    except MemoryError as error:
                        slices = slices / 2
                        print('Not enought RAM, new slices = {}'.format(slices))

                    else:
                        test = False
            else:
                print('Class prediction was successful without slicing!')

            try:
                class_prediction = np.concatenate(class_preds, axis=0)
            except NameError:
                print('No slicing was necessary!')

            class_prediction = class_prediction.reshape(img[:, :, 0].shape)
            return class_prediction
        except ValueError as e:
            print(e)
            print("Could not Predict Classification Data...")

    def plot_class_prediction(self,img,class_prediction):
        try:
            mask = np.copy(img[:, :, 0])
            mask[mask > 0.0] = 1.0  # all actual pixels have a value of 1.0

            fig = plt.figure(figsize=(7, 6))
            fig.suptitle('XTitleX data', fontsize=14)
            plt.imshow(mask)

            class_prediction.astype(np.float16)
            class_prediction_ = class_prediction * mask
            #
            plt.subplot(121)
            plt.imshow(class_prediction, cmap=plt.cm.Spectral)
            plt.title('classification unmasked')

            plt.subplot(122)
            plt.imshow(class_prediction_, cmap=plt.cm.Spectral)
            plt.title('classification masked')
            #plt.show()
        except ValueError as e:
            print(e)
            print("Could not Plot Classification Data")

    def save_result_image(self, img_ds,img,class_prediction,classification_image):
        try:
            mask = np.copy(img[:, :, 0])
            mask[mask > 0.0] = 1.0

            class_prediction.astype(np.float16)
            class_prediction_ = class_prediction * mask

            cols = img.shape[1]
            rows = img.shape[0]

            class_prediction_.astype(np.float16)

            driver = gdal.GetDriverByName("gtiff")
            outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16)
            outdata.SetGeoTransform(img_ds.GetGeoTransform())  ##sets same geotransform as input
            outdata.SetProjection(img_ds.GetProjection())  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(class_prediction_)
            outdata.FlushCache()  ##saves to disk!!
            print('Image saved to: {}'.format(classification_image))
        except ValueError as e:
            print(e)
            print("Could not save Image after Classification")

    def get_classification_labels(self,roi):
        try:
            labels = np.unique(roi[roi > 0])
            return labels
        except ValueError as e:
            print(e)
            print("Failed to load labels")

    def validation_data_sample(self,roi_v):
        try:
            n_val = (roi_v > 0).sum()
            return n_val
        except ValueError as e:
            print(e)


    def validation_labels(self,roi_v):
        try:
            labels_v = np.unique(roi_v[roi_v > 0])
            return labels_v
        except ValueError as e:
            print(e)
            print("Validation Label Error")

    def validation_processing(self,roi_v,class_prediction,roi,doc,model_path,Image_savePath,dir_path):
        try:
            rt = ReportModule()
            self.validation_data_sample(roi_v)
            self.validation_labels(roi_v)

            labels = self.get_classification_labels(roi)

            X_v = class_prediction[roi_v > 0]
            y_v = roi_v[roi_v > 0]

           # print('Our X matrix is sized: {sz_v}'.format(sz_v=X_v.shape))


            # Cross-tabulate predictions
            # confusion matrix
            convolution_mat = pd.crosstab(y_v, X_v, margins=True)

            #print(convolution_mat)

            # if you want to save the confusion matrix as a CSV file:
            # savename = 'C:\\save\\to\\folder\\conf_matrix_' + str(est) + '.csv'
            # convolution_mat.to_csv(savename, sep=';', decimal = '.')

            # information about precision, recall, f1_score, and support:
            # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
            target_names = list()
            for name in range(1, (labels.size) + 1):
                target_names.append(str(name))
            sum_mat = classification_report(y_v, X_v, target_names=target_names)

            sum_mat = classification_report(y_v, X_v, target_names=target_names, output_dict=True)
            df_sum_mat = pd.DataFrame(sum_mat).transpose()
            #
            #print(df_sum_mat)
            for col in df_sum_mat.columns:
                print(col)

                # Overall Accuracy (OAA)
            score = ('OAA = {} %'.format(accuracy_score(y_v, X_v) * 100))

            doc = rt.Clf_prepare_section3(doc,X_v,y_v,convolution_mat,df_sum_mat,score,roi_v,model_path,Image_savePath,dir_path)
            return doc
        except ValueError as e:
            print(e)
            print("Could not Validate after classification")

    def save_model(self,model,file_name):
        try:
            joblib.dump(model, file_name)
        except ValueError as e:
            print(e)
            print("Can not save your model")



