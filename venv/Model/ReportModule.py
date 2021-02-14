try:
    import sys,logging
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Table,TableStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg
    from io import BytesIO
    from datetime import date
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    from reportlab.platypus.tables import Table
    import seaborn as sns
    import os
    import pandas as pd
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict
    from sklearn import preprocessing
    # sys.path.append(r"F:\Work\Maptor\venv\HelpingModel")
    from PLSR_SSS_Helper import PLSR_SSS_Helper
    from RFHelper import RFHelper

    from osgeo import gdal, ogr, gdal_array  # I/O image data
    import numpy as np  # math and array handling
    import matplotlib.pyplot as plt  # plot figures
    import pandas as pd  # handling large data as table sheets
    from joblib import dump, load
    from operator import itemgetter

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict
except Exception as e:
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)


class ReportModule():
  
    def build_doc(self,path,mode):
        try:
            doc = canvas.Canvas(path)
            doc.setLineWidth(.3)
            doc.setFont('Helvetica-Bold', 16)
            doc.drawString(30, 810, mode)
            doc.setFont('Helvetica', 12)
            doc.drawString(30, 793, 'Software Maptor v1.4')
            doc.drawString(30, 779, "Developer: Florian Beyer & Qasim Usmani")
            doc.drawString(30, 765, 'Support: florian.beyer@uni-rostock.de')
            today = date.today()
            doc.drawString(30, 751, "Date: " + str(today))
            doc.drawString(30, 737, '')
            doc.line(20, 735, 570, 735)
            return doc
        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)

    def Clf_prepare_report(self,doc,img,roi,importance,table_M,trees,ob_score,class_prediction,ValidationData,attribute,dir_path):
        try:
            os.mkdir(dir_path+"/Graphs")

            path = dir_path + "/Graphs/"

            fig = plt.figure(figsize=(6, 6))
            plt.subplot(221)
            plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            plt.title('RS Image - first band')

            plt.subplot(222)
            plt.imshow(class_prediction, cmap=plt.cm.Spectral)
            plt.title('Classification result')

            plt.subplot(223)
            plt.imshow(roi, cmap=plt.cm.Spectral)
            plt.title('Training Data')

            plt.subplot(224)
            plt.imshow(ValidationData, cmap=plt.cm.Spectral)
            plt.title('Validation Data')

            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)

            imgdata.seek(0)  # rewind the data

            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 30, 210)

            fig.clf()

            plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            plt.title('RS Image - first band')
            plt.savefig(path + "'RS_Image.png", dpi=300)
            plt.clf()

            plt.imshow(class_prediction, cmap=plt.cm.Spectral)
            plt.title('Classification result')
            plt.savefig(path + "Classification_result.png", dpi=300)
            plt.clf()

            plt.imshow(roi, cmap=plt.cm.Spectral)
            plt.title('Training Data')
            plt.savefig(path + "Training_Data.png", dpi=300)
            plt.clf()

            plt.imshow(ValidationData, cmap=plt.cm.Spectral)
            plt.title('Validation Data')
            plt.savefig(path + "Validation_Data.png", dpi=300)
            plt.clf()

            doc.showPage()

            # doc.line(20, 810, 570, 810)
            # doc.setLineWidth(.3)
            # doc.setFont('Helvetica-Bold', 14)
            # doc.drawString(30, 790, 'Section 1: General Information and Training')
            # doc.drawString(30, 770, "Section 2: Validation")
            # doc.line(20, 750, 570, 750)

            n_samples = (roi > 0).sum()

            # doc.drawString(30, 730, "Section :1")

            doc.setFont('Helvetica', 12)

            doc.drawString(30, 710, 'Image Extent: ' + str(img.shape[0]) + " x " + str(img.shape[1]))
            doc.drawString(30, 690, 'Number of Bands: ' + str(len(importance)))

            doc.drawString(30, 670, 'Field Name (shape file) of your classes: ' + attribute)

            doc.drawString(30, 650, 'Random Forest Training: ')
            doc.drawString(30, 630, 'Number of Trees: ' + str(trees))
            doc.drawString(30, 610, 'Number of training Pixel: ' + str(n_samples))

            # What are our classification labels?
            labels = np.unique(roi[roi > 0])
            # print('Number of classes :'+ labels.size)
            doc.drawString(30, 590, 'Number of classes :' + str(labels.size))
            doc.drawString(30, 570, 'Out-Of-Bag prediction of accuracy: ' + str(ob_score))

            # doc.showPage()

            X = img[roi > 0, :]
            y = roi[roi > 0]
            data = ["Band","Importance"]
            data= [(k, v) for k, v in importance.items()]
            data2 = data[:]

            # dummy = [['00', '01'],
            #         ['10', '11'],
            #         ['20', '21'],
            #         ['30', '31'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['10', '11'],
            #         ['20', '21'],
            #         ['30', '31'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['00', '01'],
            #         ['10', '11'],
            #         ['20', '21'],
            #         ['30', '31'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['10', '11'],
            #         ['20', '21'],
            #         ['30', '31'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #          ['00', '01'],
            #          ['10', '11'],
            #          ['20', '21'],
            #          ['30', '31'],
            #          ['20', '21'],
            #          ['20', '21'],
            #          ['20', '21'],
            #          ['20', '21'],
            #          ['10', '11'],
            #          ['20', '21'],
            #          ['30', '31'],
            #          ['20', '21'],
            #          ['20', '21'],
            #          ['20', '21'],
            #         ['00', '01'],
            #         ['10', '11'],
            #         ['20', '21'],
            #         ['30', '31'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['10', '11'],
            #         ['20', '21'],
            #         ['30', '31'],
            #         ['20', '21'],
            #         ['20', '21'],
            #         ['20', '21']]
            #
            # data.extend(dummy)

            data2.sort(key=lambda x: x[1], reverse=True)
            data.insert(0, ("Band", "Importance"))
            data2.insert(0, ("Band", "Importance"))

            fig = plt.figure(figsize=(6, 6))
            sn.heatmap(table_M, annot=True, cmap="BuPu", fmt='g')
            #plt.show()
            imgdata = BytesIO()
            fig.savefig(path+"pred_acc_training.png")
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            imgdata.seek(0)  # rewind the data
            drawing = svg2rlg(imgdata)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30,520,"Convolution matrix (prediction accuracy of the training data):")
            renderPDF.draw(drawing, doc, 30, 30)
            doc.showPage()



            doc.drawString(30, 700, "Band importance (left: ordered by band number | right: ordered by importance):")
            if (len(data) > 21):
                chunks = (self.chunks(data, 20))
                ordered_chunks = (self.chunks(data2, 20))
                iterationNumer = 0
                for unorder, ordered in zip(chunks, ordered_chunks):
                    if iterationNumer != 0:
                        unorder.insert(0, ("Band", "Importance"))
                        ordered.insert(0, ("Band", "Importance"))
                    unordered_table = Table(unorder)
                    ordered_table = Table(ordered)
                    unordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    ordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))


                    unordered_table.wrapOn(doc, 60, 200)
                    unordered_table.drawOn(doc, 60, 200)

                    ordered_table.wrapOn(doc, 280, 200)
                    ordered_table.drawOn(doc, 280, 200)
                    # else:
                    #     unordered_table.wrapOn(doc, 60, 400)
                    #     unordered_table.drawOn(doc, 60, 400)
                    #
                    #     ordered_table.wrapOn(doc, 280, 400)
                    #     ordered_table.drawOn(doc, 280, 400)

                    columns = [""]
                    ordered_table = Table(columns)
                    unordered_table = Table(columns)
                    doc.showPage()
                    iterationNumer += 1
            else:
                table = Table(data)
                table2 = Table(data2)

                table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table.wrapOn(doc, 60, 400)
                table.drawOn(doc, 60, 400)

                table2.wrapOn(doc, 280, 400)
                table2.drawOn(doc, 280, 400)

                doc.showPage()

            #print(trees)
            #print(table)
            #print(importance)
            return doc
        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)


    def Clf_prepare_section3(self,doc,X_v,y_v,convolution_mat,df_sum_mat,score,roi_v,model_path,Image_savePath,dir_path):
        try:
            #print(Image_savePath+"here.................!!")
            dir_path +="/Graphs/"
            n_samples = (roi_v > 0).sum()
            #print("Number of validation Pixels: "+ str(n_samples))
            doc.setFont('Helvetica-Bold', 12)
            doc.drawString(30, 780, "Number of validation Pixels: "+ str(n_samples))
            doc.setFont('Helvetica-Bold', 12)
            #print(score)
            doc.drawString(30, 760, str(score))
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 720, "Convolution matrix (prediction accuracy of the validation data")


            convolution_mat.index = convolution_mat.index.rename('truth')
            convolution_mat.columns = convolution_mat.columns.rename('predict')

            del convolution_mat['All']
            convolution_mat = convolution_mat.drop('All', axis=0)

            fig = plt.figure(figsize=(6, 5))
            sn.heatmap(convolution_mat, annot=True, cmap="BuPu", fmt='g')
            # plt.show()

            imgdata = BytesIO()
            fig.savefig(dir_path + "pred_acc_validation.png")
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)

            imgdata.seek(0)  # rewind the data

            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 40, 300)
            fig.clf()

            df_sum_mat = df_sum_mat.round(3)

            accuracy = df_sum_mat.iloc[-3:]
            df_sum_mat = df_sum_mat.head(-3)
            r, c = df_sum_mat.shape
            seq = list(range(1, r+1))
            df_sum_mat.insert(0,"band",seq)

            data = np.array(df_sum_mat).tolist()
            data.insert(0,("class","precision","recall","f1-score","support"))

            t1 = Table(data)

            t1.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

            doc.showPage()

            doc.drawString(30, 760,"Precision, recall, F-measure and support for each class:")

            t1.wrapOn(doc, 30, 500)
            t1.drawOn(doc, 90, 500)

            # acc_col_name = ['accuracy','macro avg','weighted avg']
            # accuracy.insert(0,".",acc_col_name)
            # acc_data = np.array(accuracy).tolist()
            # t2 = Table(acc_data)
            #
            # t2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            #                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))
            # t2.wrapOn(doc, 30, 100)
            # t2.drawOn(doc, 330, 100)
            doc.setFont('Helvetica', 8)
            doc.drawString(20, 370, "Image Saved to Path: "+Image_savePath)
            # if model_path!="/":
            #     doc.drawString(30, 20, "Model Saved to Path: Model not Save")
            # else:
            doc.drawString(20, 350, "Model Saved to Path:"+ model_path)

            return doc
        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)


    def make_rf_reg_report(self,doc,img,RFHelper,dir_path):
        # Display images
        try:
            os.mkdir(dir_path + "/Graphs")

            path = dir_path + "/Graphs/"

            width, height = A4
            print("here........."+RFHelper.reportpath)
            print(RFHelper.prediction_map)
            print(RFHelper.modelsavepath)
            print(RFHelper.img_path)
            print(RFHelper.train_data)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 700,"DIRECTORIES:")

            doc.setFont('Helvetica', 7)
            doc.drawString(30, 680, "Remote Sensing Image: " + RFHelper.img_path)
            doc.drawString(30, 660, "Shape file: " + RFHelper.train_data)
            doc.drawString(30, 640, "Report Save Path: "+ RFHelper.reportpath)
            doc.drawString(30, 620, "Regression image saved to: " +RFHelper.prediction_map)
            if RFHelper.modelsavepath !='/':
                doc.drawString(30, 600, "Model saved to: "+RFHelper.modelsavepath)
            else:
                doc.drawString(30, 600, "Model saved to: Model not saved")

            doc.line(20, 580, 570, 580)

            if img.shape[0] > img.shape[1]:
                fig = plt.figure(figsize=(5, 4))
                plt.subplot(121)
                print(img.shape)
                plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                plt.title('RS image - first band')
                plt.subplot(122)
                plt.imshow(RFHelper.training, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
                plt.title('Training Image')
                #plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 50, 160)
            # else:
            #     renderPDF.draw(drawing, doc, 50, 250)
                fig.clf()
                plt.close(fig)
                fig = plt.figure(figsize=(5, 4))
                plt.imshow(RFHelper.prediction, cmap=plt.cm.Spectral)
                plt.title('Prediction')
               # plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 400, 160)
                fig.clf()
                plt.close(fig)
                doc.showPage()


            if img.shape[0]<=  img.shape[1]:
                fig = plt.figure(figsize=(6, 4))
                plt.subplot(121)
                print(img.shape)
                plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                plt.title('RS image - first band')
                plt.subplot(122)
                plt.imshow(RFHelper.training, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
                plt.title('Training Image')
                #plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 50, 360)
                # else:
                #     renderPDF.draw(drawing, doc, 50, 250)
                fig.clf()
                plt.close(fig)
                fig = plt.figure(figsize=(5, 4))
                plt.imshow(RFHelper.prediction, cmap=plt.cm.Spectral)
                plt.title('Prediction')
               # plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 70, 60)
                fig.clf()
                plt.close(fig)
                doc.showPage()

            plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            plt.title('RS image - first band')
            plt.savefig(path + "RS image-first_band.png", dpi=300)
            plt.clf()

            plt.imshow(RFHelper.training, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
            plt.title('Training Image')
            plt.savefig(path + "Training_Image.png", dpi=300)
            plt.clf()

            plt.imshow(RFHelper.prediction, cmap=plt.cm.Spectral)
            plt.title('Prediction')
            plt.savefig(path + "Prediction.png", dpi=300)
            plt.clf()

            # doc.line(20,800, 570, 800)
            # doc.setLineWidth(.3)
            # doc.setFont('Helvetica-Bold', 14)
            # doc.drawString(30, 780, 'Section 1: General Information and Training')
            # doc.drawString(30, 765, "Section 2: Validation")
            # doc.line(20, 750, 570, 750)



            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 730, 'Section 1:')
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 710,"Image extent: "+RFHelper.helper.imageExtent + "(Rows x Columns)")
            doc.drawString(30, 690, "Number of Bands: "+ str(RFHelper.helper.BandNumber))
            doc.drawString(30, 670, "Field name (shape file) of your classes: "+RFHelper.helper.FieldName)
            doc.setFont('Helvetica', 14)
            doc.drawString(30,650," Random Forrest Training")
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 630, " Number of Tress: "+ str(RFHelper.helper.TreesNo))
            doc.drawString(30, 610, " Number of samples: "+ str(RFHelper.helper.SampleNo))
            doc.drawString(30, 590, " Split size for Test: "+str(RFHelper.helper.SplitSize))
            doc.drawString(30, 570, " Training samples: "+str(RFHelper.helper.TrainingSample))
            doc.drawString(30, 550, " Test sample: "+str(RFHelper.helper.TestSample))

            doc.drawString(30, 530, "Co efficient of determination R^2 of the prediction : "+RFHelper.helper.Coeff)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 510, "left: ordered by band number| right: ordered by importance ")

            data = ["Band", "Importance"]
            data = [(k, v) for k, v in RFHelper.helper.BandImportance.items()]

            # dummy = [('Band',1.33),('Band',5),('Band',4),('Band',2),('Band',6),('Band',8),('Band',43),('Band',3),('Band',113),('Band',233),('Band',13),('Band',133)]
            # data.extend(dummy)


            data2 = data[:]
            data2.sort(key=lambda x: x[1], reverse=True)

            data.insert(0, ("Band", "Importance"))
            data2.insert(0, ("Band", "Importance"))

            datalen = len(data)
            print((datalen))

            if(len(data)>21):
                chunks = (self.chunks(data,20))
                ordered_chunks = (self.chunks(data2, 20))
                iterationNumer = 0
                for unorder,ordered in zip(chunks,ordered_chunks):
                    if iterationNumer != 0:
                        unorder.insert(0, ("Band", "Importance"))
                        ordered.insert(0, ("Band", "Importance"))
                    unordered_table = Table(unorder)
                    ordered_table = Table(ordered)
                    unordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    ordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    if iterationNumer == 0:
                        unordered_table.wrapOn(doc, 60, 100)
                        unordered_table.drawOn(doc, 60, 100)

                        ordered_table.wrapOn(doc, 280, 100)
                        ordered_table.drawOn(doc,280, 100)
                    else:
                        unordered_table.wrapOn(doc, 60, 400)
                        unordered_table.drawOn(doc, 60, 400)

                        ordered_table.wrapOn(doc, 280, 400)
                        ordered_table.drawOn(doc, 280, 400)


                    columns = [""]
                    ordered_table = Table(columns)
                    unordered_table = Table(columns)
                    doc.showPage()
                    iterationNumer += 1

            else:
                table = Table(data)
                table2 = Table(data2)

                table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                           ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))


                table.wrapOn(doc, 0,0)
                table.drawOn(doc, 60, 100)

                table2.wrapOn(doc, 60, 100)
                table2.drawOn(doc, 280, 100)
                doc.showPage()


            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 790, 'Section 2:')
            doc.setFont('Helvetica', 14)

            doc.drawString(30, 770, "n of the data: " + str(RFHelper.helper.n_testdata))
            doc.drawString(30, 750, "Mean of the Variable : " + str(RFHelper.helper.meanvariable))
            doc.drawString(30, 730, "Standard Deviation of the Variable: " + str(round(RFHelper.helper.stdDev,2)))
            doc.drawString(30, 710, "----------------------------------------")
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 670, " Mean Absolute Error : " + str(round(RFHelper.helper.absError,2)))
            doc.drawString(30, 650, " Mean Squared Error:" + str(round(RFHelper.helper.meanSqError,2)))
            doc.drawString(30, 630, " RMSE:" + str(round(RFHelper.helper.RMSE,2)))
            x = str(round(RFHelper.helper.MAPE,2))

            doc.drawString(30, 600, " Mean absoulute percentage error(MAPE)/ Accuracy :"  + x + " %")
            doc.drawString(30, 560, "----------------------------------------")

            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 540, "Co efficient of determination R^2 of the prediction : " + str(round(RFHelper.helper.coeffR,2)))

            fig, ax = plt.subplots()
            ax.scatter(RFHelper.helper.testlabels, RFHelper.helper.pred_test_ds)
            ax.plot([RFHelper.helper.testlabels.min(), RFHelper.helper.testlabels.max()], [RFHelper.helper.testlabels.min(), RFHelper.helper.testlabels.max()], 'k--', lw=1)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')

            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            plt.savefig(path + "r2_Prediction.png", dpi=300)

            imgdata.seek(0)  # rewind the data

            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 30, 110)
            fig.clf()
            plt.close(fig)

            return doc
        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)

    def make_plsr_sss_report(self, doc, img, dir_path, PLSR_SSS_Helper):

        # self, doc, img, TrainingData, X, mse, msemin, component, y, y_c, y_cv, attribute, importance, prediction, dir_path, reportpath, prediction_map, modelsavepath, img_path, trn_path
        try:

            # reportparameters = PLSR_SSS_Helper()
            # reportparameters = helpermodel

            print(PLSR_SSS_Helper)
            print(type(PLSR_SSS_Helper))

            os.mkdir(dir_path + "/Graphs")

            path = dir_path + "/Graphs/"

            width, height = A4
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 705, "DIRECTORIES:")

            doc.setFont('Helvetica', 10)
            doc.drawString(30, 680, "Remote Sensing Image: " + str(PLSR_SSS_Helper.img_path))
            doc.drawString(30, 660, "Shape file: " + str(PLSR_SSS_Helper.tran_path))
            doc.drawString(30, 640, "Report Save Path: " + str(PLSR_SSS_Helper.reportpath))
            doc.drawString(30, 620, "Regression image saved to: " + str(PLSR_SSS_Helper.prediction_map))
            # if modelsavepath != '/':
            doc.drawString(30, 600, "Model saved to: " + str(PLSR_SSS_Helper.modelsavepath))
            # else:
            #     doc.drawString(30, 600, "Model saved to: Model not saved")

            doc.line(20, 580, 570, 580)

            mask = np.copy(img[:, :, 0])
            mask[mask > 0.0] = 1.0  # all actual pixels have a value of 1.0
            # plt.imshow(mask)
            prediction_ = PLSR_SSS_Helper.prediction * mask

            if img.shape[0] > img.shape[1]:
                fig = plt.figure(figsize=(5, 4))
                plt.subplot(121)
                plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                roi_positions = np.where(PLSR_SSS_Helper.train_data > 0)
                plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
                plt.title('first RS band and sample points', fontsize=8)

                plt.subplot(122)
                plt.imshow(prediction_, cmap=plt.cm.Spectral, vmin=PLSR_SSS_Helper.y.min(), vmax=PLSR_SSS_Helper.y.max())
                plt.title('Prediction: ' + PLSR_SSS_Helper.attribute, fontsize=8)
                plt.colorbar()
                imgdata = BytesIO()
                plt.savefig(path + "RS image-first_band.png", dpi=300)
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 50, 160)

                fig.clf()
                plt.close(fig)
                doc.showPage()

            if img.shape[0] <= img.shape[1]:
                fig = plt.figure(figsize=(5, 4))
                plt.subplot(121)
                plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                roi_positions = np.where(PLSR_SSS_Helper.train_data > 0)
                plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
                plt.title('first RS band and sample points')

                plt.subplot(122)
                plt.imshow(prediction_, cmap=plt.cm.Spectral, vmin=PLSR_SSS_Helper.y.min(), vmax=PLSR_SSS_Helper.y.max())
                plt.title('Prediction')
                plt.colorbar()
                imgdata = BytesIO()
                plt.savefig(path + "RS image-first_band.png", dpi=300)
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 100, 160)

                fig.clf()
                plt.close(fig)
                doc.showPage()

            n_samples = (PLSR_SSS_Helper.train_data > 0).sum()
            # doc.drawString(30, 720, 'We have {n} training samples'.format(n=n_samples))

            # Subset the image dataset with the training image = X
            # Mask the classes on the training dataset = y
            # These will have n_samples rows

            features = pd.DataFrame(PLSR_SSS_Helper.X)

            band_names = []
            for i in range(PLSR_SSS_Helper.X.shape[1]):
                # for i in range(0,2500):
                nband = "Band_" + str(i + 1)
                band_names.append(nband)

            features.columns = band_names
            print(features.shape)

            doc.line(20, 810, 570, 810)
            doc.setLineWidth(.3)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 785, 'Section : General Information and Training')
            doc.line(20, 770, 570, 770)
            doc.setFont('Helvetica', 10)
            doc.drawString(30, 700,
                           'The Image Extend: ' + str(img.shape[0]) + " x " + str(img.shape[1]) + " (Rows x Columns)")
            doc.drawString(30, 680, 'The number of bands is: ' + str(features.shape[1]))
            # doc.drawString(30, 680,'The shape of our features is: '+str(features.shape))
            doc.drawString(30, 660, 'Selected Attribute: ' + str(PLSR_SSS_Helper.attribute))
            doc.drawString(30, 640, 'The number of Sample is: ' + str(features.shape[0]))
            doc.drawString(30, 620, '---------------------------------------------------------')

            #
            features['value'] = PLSR_SSS_Helper.y
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

            # doc.drawString(30, 620, 'Training Features Shape: ' + str(features.shape))
            # doc.drawString(30, 600, 'Training Labels Shape: ' + str(labels.shape))

            # return [doc, labels, features]

            #
            #  doc.showPage()
            suggested_comp = PLSR_SSS_Helper.msemin + 1
            print("Suggested number of components: ", suggested_comp)
            doc.drawString(30, 560, "Selected number of PLS components: " + str(suggested_comp))

            fig = plt.figure(figsize=(5, 4))
            with plt.style.context(('ggplot')):
                plt.plot(PLSR_SSS_Helper.component, np.array(PLSR_SSS_Helper.mse), '-v', color='blue', mfc='blue')
                plt.plot(PLSR_SSS_Helper.component[PLSR_SSS_Helper.msemin], np.array(PLSR_SSS_Helper.mse)[PLSR_SSS_Helper.msemin], 'P',
                         ms=10, mfc='red')
                plt.xlabel('Number of PLS components')
                plt.ylabel('MSE')
                plt.title('PLSR MSE vs. Components')
                plt.xlim(left=-1)
                plt.savefig(path + "PLSR MSE vs. Components.png", dpi=300)

            # plt.show()
            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            imgdata.seek(0)  # rewind the data
            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 50, 150)
            fig.clf()
            plt.close(fig)
            doc.showPage()

            score_c = r2_score(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_c)
            score_cv = r2_score(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_cv)

            # Calculate mean squared error for calibration and cross validation
            mse_c = mean_squared_error(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_c)
            mse_cv = mean_squared_error(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_cv)

            #   print("2 HERE!!!!!! mse_cv"+str(mse_cv))

            print('R2 calib: %5.3f' % score_c)
            doc.drawString(30, 730, 'R2 of the training: %5.3f' % score_c)
            print('R2 LOOCV: %5.3f' % score_cv)
            doc.drawString(30, 710, 'R2 LOOCV: %5.3f' % score_cv)
            print('MSE calib: %5.3f' % mse_c)
            doc.drawString(30, 690, 'MSE of the training: %5.3f' % mse_c)
            print('MSE LOOCV: %5.3f' % mse_cv)
            doc.drawString(30, 670, 'MSE LOOCV: %5.3f' % mse_cv)

            imp = {}
            for i in range(features.shape[1]):
                print('Band {}: {}'.format(i + 1, round(PLSR_SSS_Helper.importance[i], 2)))
                imp['Band{}'.format(i + 1)] = round(PLSR_SSS_Helper.importance[i], 2)

            data = [(k, v) for k, v in imp.items()]

            # dummy = [('Band',1.33),('Band',5),('Band',4),('Band',2),('Band',6),('Band',8),('Band',43),('Band',3),('Band',113),('Band',233),('Band',13),('Band',133)]
            # data.extend(dummy)

            data2 = data[:]

            data2.sort(key=lambda x: x[1], reverse=True)

            data.insert(0, ("Band", "Importance"))
            data2.insert(0, ("Band", "Importance"))
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 510, "Band importance (left: ordered by band number | right: ordered by importance):")
            doc.setFont('Helvetica', 10)
            if (len(data) > 21):
                chunks = (self.chunks(data, 20))
                ordered_chunks = (self.chunks(data2, 20))
                iterationNumer = 0
                for unorder, ordered in zip(chunks, ordered_chunks):
                    if iterationNumer != 0:
                        unorder.insert(0, ("Band", "Importance"))
                        ordered.insert(0, ("Band", "Importance"))
                    unordered_table = Table(unorder)
                    ordered_table = Table(ordered)
                    unordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    ordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    if iterationNumer == 0:
                        unordered_table.wrapOn(doc, 60, 100)
                        unordered_table.drawOn(doc, 60, 100)

                        ordered_table.wrapOn(doc, 280, 100)
                        ordered_table.drawOn(doc, 280, 100)
                    else:
                        unordered_table.wrapOn(doc, 60, 400)
                        unordered_table.drawOn(doc, 60, 400)

                        ordered_table.wrapOn(doc, 280, 400)
                        ordered_table.drawOn(doc, 280, 400)

                    columns = [""]
                    ordered_table = Table(columns)
                    unordered_table = Table(columns)
                    doc.showPage()
                    iterationNumer += 1

            else:
                table = Table(data)
                table2 = Table(data2)

                table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                           ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table.wrapOn(doc, 0, 0)
                table.drawOn(doc, 60, 100)

                table2.wrapOn(doc, 60, 100)
                table2.drawOn(doc, 280, 100)
                doc.showPage()

            doc.line(20, 810, 570, 810)
            doc.setLineWidth(.3)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 785, 'Section : Validation')
            doc.line(20, 770, 570, 770)

            z = np.polyfit(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_c, 1)
            with plt.style.context(('ggplot')):
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(PLSR_SSS_Helper.y_c, PLSR_SSS_Helper.y, c='red', edgecolors='k')
                # Plot the best fit line
                ax.plot(np.polyval(z, PLSR_SSS_Helper.y), PLSR_SSS_Helper.y, c='blue', linewidth=1)
                # Plot the ideal 1:1 line
                ax.plot(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y, color='green', linewidth=1)
                plt.title('$R^{2}$ (CV): ' + str(score_cv))
                plt.xlabel('Predicted')
                plt.ylabel('Measured')
                plt.legend(['best fit line', 'ideal 1:1 line', 'samples'])
                imgdata = BytesIO()
                plt.savefig(path + "R2CV", dpi=300)
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 50, 300)
                fig.clf()
                plt.close(fig)

            # In[16]:

            # Calculate the absolute errors
            errors = abs(PLSR_SSS_Helper.y - PLSR_SSS_Helper.y_cv)
            # Print out the mean absolute error (mae)

            # Print out the mean absolute error (mae)
            doc.setFont('Helvetica', 10)
            doc.drawString(30, 270, '----------------------------')
            print('-------------')
            print('n of the test data: {}'.format(len(labels)))
            doc.drawString(30, 250, 'n of the test data: ' + str((len(labels))))

            print('Mean of the variable: {:.2f}'.format(np.mean(labels)))

            doc.drawString(30, 230, 'Mean of the Samples: ' + str(round((np.mean(labels)), 2)))

            print('Standard deviation of the variable: {:.2f}'.format(np.std(labels)))
            doc.drawString(30, 210, 'Standard deviation of the Samples: ' + str(round(np.std(labels), 2)))

            print('-------------')
            doc.drawString(30, 190, '----------------------------')
            print('Mean Absolute Error: {:.2f}'.format(np.mean(errors)))
            doc.drawString(30, 170, 'MAE: ' + str(round(np.mean(errors), 2)))

            print('Mean squared error: {:.2f}'.format(mse_cv))
            doc.drawString(30, 150, 'MSE: ' + str(round(mse_cv, 2)))

            print('RMSE: {:.2f}'.format(np.sqrt(mse_cv)))
            doc.drawString(30, 130, 'RMSE: ' + str(round(np.sqrt(mse_cv), 2)))

            RPD = np.std(labels) / np.sqrt(mse_cv)

            print('RPD: {:.2f} | How often does RMSE of Prediction fit in the Standard Deviation of the samples'.format(
                RPD))
            doc.drawString(30, 110, 'RPD: ' + str(
                round(RPD, 2)) + "| How often does RMSE of Prediction fit in the Standard Deviation of the samples")

            mape = 100 * (errors / labels)
            # Calculate and display accuracy
            accuracy = 100 - np.mean(mape)
            print('mean absolute percentage error (MAPE): {:.2f} %'.format(np.mean(mape)))
            doc.drawString(30, 90, 'Mean Absolute Percentage Error (MAPE): ' + str(round(np.mean(mape), 2)) + " %")

            print('accuracy (100 % - mape): {:.2f} %'.format(accuracy))
            doc.drawString(30, 70, 'Accuracy (100 % - MAPE): ' + str(round(accuracy, 2)) + " %")
            print('-------------')
            doc.drawString(30, 50, '----------------------------')

            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination r²: {:.2f}'.format(r2_score(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_cv)))
            doc.drawString(30, 30,
                           'Coefficient of determination r²: ' + str(round(r2_score(PLSR_SSS_Helper.y, PLSR_SSS_Helper.y_cv), 2)))
            #
            return doc

        except ValueError as e:
            logging.error("Exception occurred", exc_info=True)
            print(e)

    def make_plsr_lds_report(self,doc,dir_path,PLSR_LDS_Helper):

        os.mkdir(dir_path+"/Graphs")
        path = dir_path +"/Graphs/"
        doc.setFont('Helvetica-Bold', 14)
        doc.drawString(30, 705, "DIRECTORIES:")

        doc.setFont('Helvetica', 10)
        doc.drawString(30, 680, "Remote Sensing Image: " + str(PLSR_LDS_Helper.img_path))
        doc.drawString(30, 660, "Shape file: " + str(PLSR_LDS_Helper.tran_path))
        doc.drawString(30, 640, "Report Save Path: " + str(PLSR_LDS_Helper.reportpath))
        doc.drawString(30, 620, "Regression image saved to: " + str(PLSR_LDS_Helper.prediction_map))
        # if modelsavepath != '/':
        doc.drawString(30, 600, "Model saved to: " + str(PLSR_LDS_Helper.modelsavepath))

        doc.line(20, 580, 570, 580)

        mask = np.copy(PLSR_LDS_Helper.img[:, :, 0])
        mask[mask > 0.0] = 1.0  # all actual pixels have a value of 1.0
        prediction_ = PLSR_LDS_Helper.prediction * mask


        if PLSR_LDS_Helper.img.shape[0] > PLSR_LDS_Helper.img.shape[1]:
            fig = plt.figure(figsize=(5, 4))
            plt.subplot(121)
            print(PLSR_LDS_Helper.img.shape)
            plt.imshow(PLSR_LDS_Helper.img[:, :, 0], cmap=plt.cm.Greys_r)
            roi_positions = np.where(PLSR_LDS_Helper.train_data > 0)
            plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
            plt.title('RS image - first band')
            plt.subplot(122)
            plt.imshow(PLSR_LDS_Helper.train_data, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
            plt.title('Training Image')
            # plt.show()
            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            imgdata.seek(0)  # rewind the data
            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 50, 160)
            # else:
            #     renderPDF.draw(drawing, doc, 50, 250)
            fig.clf()
            plt.close(fig)
            fig = plt.figure(figsize=(5, 4))
            plt.imshow(PLSR_LDS_Helper.prediction, cmap=plt.cm.Spectral)
            plt.colorbar()
            plt.title('Prediction')
            # plt.show()
            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            imgdata.seek(0)  # rewind the data
            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 400, 160)
            fig.clf()
            plt.close(fig)
            doc.showPage()

        if PLSR_LDS_Helper.img.shape[0] <= PLSR_LDS_Helper.img.shape[1]:
            fig = plt.figure(figsize=(5, 4))
            plt.subplot(121)
            print(PLSR_LDS_Helper.img.shape)
            plt.imshow(PLSR_LDS_Helper.img[:, :, 0], cmap=plt.cm.Greys_r)
            roi_positions = np.where(PLSR_LDS_Helper.train_data > 0)
            plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
            plt.title('RS image - first band')
            plt.subplot(122)
            plt.imshow(PLSR_LDS_Helper.train_data, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
            plt.title('Training Image')
            # plt.show()
            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            imgdata.seek(0)  # rewind the data
            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 50, 160)
            # else:
            #     renderPDF.draw(drawing, doc, 50, 250)
            fig.clf()
            plt.close(fig)
            fig = plt.figure(figsize=(5, 4))
            plt.imshow(PLSR_LDS_Helper.prediction, cmap=plt.cm.Spectral)
            plt.colorbar()
            plt.title('Prediction')
            # plt.show()
            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
            imgdata.seek(0)  # rewind the data
            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 100, 160)
            fig.clf()
            plt.close(fig)
            doc.showPage()

        roi_positions = np.where(PLSR_LDS_Helper.train_data > 0)
        plt.imshow(PLSR_LDS_Helper.img[:, :, 0], cmap=plt.cm.Greys_r)
        plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
        plt.title('RS image - first band')
        plt.savefig(path + "RS image-first_band.png", dpi=300)
        plt.clf()

        plt.imshow(PLSR_LDS_Helper.train_data, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
        plt.title('Training Image')
        plt.savefig(path + "Training_Image.png", dpi=300)
        plt.clf()

        plt.imshow(PLSR_LDS_Helper.prediction, cmap=plt.cm.Spectral)
        plt.colorbar()
        plt.title('Prediction')
        plt.savefig(path + "Prediction.png", dpi=300)
        plt.clf()

        n_samples = (PLSR_LDS_Helper.train_data > 0).sum()
        print(
            'We have {n} training samples'.format(n=n_samples))  # Subset the image dataset with the training image = X

        print('The shape of our features is:', PLSR_LDS_Helper.features.shape)
        print('The number of Spectra is:', PLSR_LDS_Helper.features.shape[0])
        print('The number of bands is:', PLSR_LDS_Helper.features.shape[1])

        doc.line(20, 810, 570, 810)
        doc.setLineWidth(.3)
        doc.setFont('Helvetica-Bold', 14)
        doc.drawString(30, 785, 'Section : General Information and Training')
        doc.line(20, 770, 570, 770)
        doc.setFont('Helvetica', 10)
        doc.drawString(30, 700,
                       'The Image Extend: ' + str(PLSR_LDS_Helper.img.shape[0]) + " x " + str(PLSR_LDS_Helper.img.shape[1]) + " (Rows x Columns)")
        doc.drawString(30, 680, 'The number of bands is: ' + str(PLSR_LDS_Helper.features.shape[1]))
        # doc.drawString(30, 680,'The shape of our features is: '+str(features.shape))
        doc.drawString(30, 660, 'Selected Attribute: ' + str(PLSR_LDS_Helper.attribute))
        doc.drawString(30, 640, 'The number of Sample is: ' + str(PLSR_LDS_Helper.features.shape[0]))
        doc.drawString(30, 620, '---------------------------------------------------------')




        # from sklearn import preprocessing
        #
        # min_max_scaler = preprocessing.MinMaxScaler()
        #
        # xscaled = min_max_scaler.fit_transform(features)
        # features_ = pd.DataFrame(xscaled)
        #
        # features_.transpose().plot(figsize=(20, 7))
        # plt.legend(bbox_to_anchor=(0.1, -0.1), loc='upper left', ncol=7)
        # plt.title('Reference Spectra')
        # plt.plot()
        #
        #
        print('Training Features Shape:', PLSR_LDS_Helper.train_features.shape)
        print('Training Labels Shape:', PLSR_LDS_Helper.train_labels.shape)
        print('Testing Features Shape:', PLSR_LDS_Helper.test_features.shape)
        print('Testing Labels Shape:', PLSR_LDS_Helper.test_labels.shape)
        #
        msemin = np.argmin(PLSR_LDS_Helper.mse)
        suggested_comp = msemin + 1
        print("Suggested number of components: ", suggested_comp)

        fig = plt.figure(figsize=(5, 4))
        with plt.style.context(('ggplot')):
            plt.plot(PLSR_LDS_Helper.component, np.array(PLSR_LDS_Helper.mse), '-v', color='blue', mfc='blue')
            plt.plot(PLSR_LDS_Helper.component[msemin], np.array(PLSR_LDS_Helper.mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLSR MSE vs. Components')
            plt.xlim(left=-1)
            plt.savefig(path + "PLSR MSE vs. Components.png", dpi=300)

        # plt.show()
        imgdata = BytesIO()
        fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
        imgdata.seek(0)  # rewind the data
        drawing = svg2rlg(imgdata)
        renderPDF.draw(drawing, doc, 50, 150)
        fig.clf()
        plt.close(fig)
        doc.showPage()

        # print(sorted_imp)
        imp = {}
        for i in range(PLSR_LDS_Helper.features.shape[1]):
            print('Band {}: {}'.format(i + 1, round(PLSR_LDS_Helper.importance[i], 2)))
            imp['Band{}'.format(i + 1)] = round(PLSR_LDS_Helper.importance[i], 2)

        data = [(k, v) for k, v in imp.items()]

        # dummy = [('Band',1.33),('Band',5),('Band',4),('Band',2),('Band',6),('Band',8),('Band',43),('Band',3),('Band',113),('Band',233),('Band',13),('Band',133)]
        # data.extend(dummy)

        data2 = data[:]

        data2.sort(key=lambda x: x[1], reverse=True)

        data.insert(0, ("Band", "Importance"))
        data2.insert(0, ("Band", "Importance"))
        doc.setFont('Helvetica-Bold', 14)
        doc.drawString(30, 710, "Band importance (left: ordered by band number | right: ordered by importance):")
        doc.setFont('Helvetica', 10)
        if (len(data) > 21):
            chunks = (self.chunks(data, 20))
            ordered_chunks = (self.chunks(data2, 20))
            iterationNumer = 0
            for unorder, ordered in zip(chunks, ordered_chunks):
                if iterationNumer != 0:
                    unorder.insert(0, ("Band", "Importance"))
                    ordered.insert(0, ("Band", "Importance"))
                unordered_table = Table(unorder)
                ordered_table = Table(ordered)
                unordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                     ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                ordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                   ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                if iterationNumer == 0:
                    unordered_table.wrapOn(doc, 60, 100)
                    unordered_table.drawOn(doc, 60, 100)

                    ordered_table.wrapOn(doc, 280, 100)
                    ordered_table.drawOn(doc, 280, 100)
                else:
                    unordered_table.wrapOn(doc, 60, 400)
                    unordered_table.drawOn(doc, 60, 400)

                    ordered_table.wrapOn(doc, 280, 400)
                    ordered_table.drawOn(doc, 280, 400)

                columns = [""]
                ordered_table = Table(columns)
                unordered_table = Table(columns)
                doc.showPage()
                iterationNumer += 1

        else:
            table = Table(data)
            table2 = Table(data2)

            table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

            table2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

            table.wrapOn(doc, 0, 0)
            table.drawOn(doc, 60, 100)

            table2.wrapOn(doc, 60, 100)
            table2.drawOn(doc, 280, 100)
            doc.showPage()

        #
        errors = abs(PLSR_LDS_Helper.predictions_test_ds - PLSR_LDS_Helper.test_labels)
        # Print out the mean absolute error (mae)

        # Print out the mean absolute error (mae)

        print('-------------')
        print('n of the test data: {}'.format(len(PLSR_LDS_Helper.test_labels)))
        print('Mean of the variable: {:.2f}'.format(np.mean(PLSR_LDS_Helper.labels)))
        print('Standard deviation of the variable: {:.2f}'.format(np.std(PLSR_LDS_Helper.labels)))
        print('-------------')
        print('Mean Absolute Error: {:.2f}'.format(np.mean(errors)))

        mse = mean_squared_error(PLSR_LDS_Helper.test_labels, PLSR_LDS_Helper.predictions_test_ds)

        print('Mean squared error: '+ str(round(mse,2)))
        print('RMSE: '+str(round(np.sqrt(mse),2)))
        #
        mape = 100 * (errors / PLSR_LDS_Helper.test_labels)
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print('mean absolute percentage error (MAPE) / Accuracy: {:.2f}'.format(accuracy), '%.')
        print('-------------')
        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination r²: {:.2f}'.format(r2_score(PLSR_LDS_Helper.test_labels, PLSR_LDS_Helper.predictions_test_ds)))
        # #
        doc.drawString(30, 785, 'n of the test data: {}'.format(len(PLSR_LDS_Helper.test_labels)))
        doc.drawString(30, 765,'Mean of the variable: {:.2f}'.format(np.mean(PLSR_LDS_Helper.labels)))
        doc.drawString(30, 745,'Standard deviation of the variable: {:.2f}'.format(np.std(PLSR_LDS_Helper.labels)))
        doc.drawString(30, 735,'------------------------------------------------------')
        doc.drawString(30, 715,'Mean Absolute Error: {:.2f}'.format(np.mean(errors)))

        mse = mean_squared_error(PLSR_LDS_Helper.test_labels, PLSR_LDS_Helper.predictions_test_ds)
        doc.drawString(30, 685,'Mean squared error: {:.2f}'.format(mse))
        doc.drawString(30, 670,'RMSE: {:.2f}'.format(np.sqrt(mse)))

        mape = 100 * (errors / PLSR_LDS_Helper.test_labels)
        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        doc.drawString(30, 655,"mean absolute percentage error (MAPE) / Accuracy: "+ str(round(accuracy,2))+" %")
        doc.drawString(30, 640,'----------------------------------------------------')


        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(PLSR_LDS_Helper.test_labels, PLSR_LDS_Helper.predictions_test_ds)
        ax.plot([PLSR_LDS_Helper.test_labels.min(), PLSR_LDS_Helper.test_labels.max()], [PLSR_LDS_Helper.test_labels.min(), PLSR_LDS_Helper.test_labels.max()], 'k--', lw=1)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        imgdata = BytesIO()
        plt.savefig(path + "R2CV", dpi=300)
        fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
        imgdata.seek(0)  # rewind the data
        drawing = svg2rlg(imgdata)
        renderPDF.draw(drawing, doc, 50, 240)
        fig.clf()
        plt.close(fig)

        # mask = np.copy(img[:, :, 0])
        # mask[mask > 0.0] = 1.0  # all actual pixels have a value of 1.0

        # plot mask

        # plt.imshow(mask)


        # mask classification an plot

        # prediction_ = prediction * mask
        #
        # plt.subplot(121)
        # plt.imshow(prediction, cmap=plt.cm.Spectral, vmax=prediction.mean() + prediction.std() * 2,
        #            vmin=prediction.mean() - prediction.std() * 2)
        # plt.title('prediction unmasked')
        #
        # plt.subplot(122)
        # plt.imshow(prediction_, cmap=plt.cm.Spectral, vmax=prediction_.mean() + prediction_.std() * 2,
        #            vmin=prediction_.mean() - prediction_.std() * 2)
        # plt.title('prediction masked')
        #
        # plt.show()
        #
        #
        # plt.subplot(121)
        # plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
        # plt.title('RS image - first band')
        #
        # plt.subplot(122)
        # plt.imshow(prediction, cmap=plt.cm.Spectral, vmax=prediction_.mean() + prediction_.std() * 2,
        #            vmin=prediction_.mean() - prediction_.std() * 2)
        # plt.colorbar()
        #
        # plt.title('Prediction')
        #
        # plt.show()
        #
        return doc

    def make_rfr_sss_report(self,doc,reportpath,dir_path,train_data,prediction_map,modelsavepath,img_path,TrainingData,img,attributes,prediction, y_c, y_cv, RFR):

        try:
            os.mkdir(dir_path + "/Graphs")

            path = dir_path + "/Graphs/"

            width, height = A4
           # print("here........."+reportpath)
            print(prediction_map)
            print(modelsavepath)
            print(img_path)
            print(train_data)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 700,"DIRECTORIES:")

            doc.setFont('Helvetica', 7)
            doc.drawString(30, 680, "Remote Sensing Image: " + img_path)
            doc.drawString(30, 660, "Shape file: " + train_data)
            doc.drawString(30, 640, "Report Save Path: "+ reportpath)
            doc.drawString(30, 620, "Regression image saved to: " + prediction_map)
            if modelsavepath !='/':
                doc.drawString(30, 600, "Model saved to: "+modelsavepath)
            else:
                doc.drawString(30, 600, "Model saved to: Model not saved")

            doc.line(20, 580, 570, 580)



            n_samples = (TrainingData > 0).sum()
            print('We have {n} training samples'.format(
                n=n_samples))  # Subset the image dataset with the training image = X

            roi_positions = np.where(TrainingData > 0)

            if img.shape[0] > img.shape[1]:
                fig = plt.figure(figsize=(5, 4))
                plt.subplot(121)
                print(img.shape)
                plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
                plt.title('RS image - first band')
                plt.subplot(122)
                plt.imshow(TrainingData, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
                plt.title('Training Image')
                #plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 50, 160)
            # else:
            #     renderPDF.draw(drawing, doc, 50, 250)
                fig.clf()
                plt.close(fig)
                fig = plt.figure(figsize=(5, 4))
                plt.imshow(prediction, cmap=plt.cm.Spectral)
                plt.colorbar()
                plt.title('Prediction')
               # plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 400, 160)
                fig.clf()
                plt.close(fig)
                doc.showPage()


            if img.shape[0]<=  img.shape[1]:
                fig = plt.figure(figsize=(6, 4))
                plt.subplot(121)
                print(img.shape)
                plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
                plt.title('RS image - first band')
                plt.subplot(122)
                plt.imshow(TrainingData, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
                plt.title('Training Image')
                #plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 50, 360)
                # else:
                #     renderPDF.draw(drawing, doc, 50, 250)
                fig.clf()
                plt.close(fig)
                fig = plt.figure(figsize=(5, 4))
                plt.imshow(prediction, cmap=plt.cm.Spectral)
                plt.colorbar()
                plt.title('Prediction')
               # plt.show()
                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                imgdata.seek(0)  # rewind the data
                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 70, 60)
                fig.clf()
                plt.close(fig)
                doc.showPage()

            plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
            plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
            plt.title('RS image - first band')
            plt.savefig(path + "RS image-first_band.png", dpi=300)
            plt.clf()

            plt.imshow(TrainingData, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
            plt.title('Training Image')
            plt.savefig(path + "Training_Image.png", dpi=300)
            plt.clf()

            plt.imshow(prediction, cmap=plt.cm.Spectral)
            plt.colorbar()
            plt.title('Prediction')
            plt.savefig(path + "Prediction.png", dpi=300)
            plt.clf()




            settings_sns = {'axes.facecolor': 'white',
                            'axes.edgecolor': '0',
                            'axes.grid': True,
                            'axes.axisbelow': True,
                            'axes.labelcolor': '.15',
                            'figure.facecolor': 'white',
                            'grid.color': '.8',
                            'grid.linestyle': '--',
                            'text.color': '0',
                            'xtick.color': '0',
                            'ytick.color': '0',
                            'xtick.direction': 'in',
                            'ytick.direction': 'in',
                            'lines.solid_capstyle': 'round',
                            'patch.edgecolor': 'w',
                            'patch.force_edgecolor': True,
                            'image.cmap': 'Greys',
                            'font.family': ['serif'],
                            'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'Bitstream Vera Sans',
                                                'sans-serif'],
                            'xtick.bottom': True,
                            'xtick.top': True,
                            'ytick.left': True,
                            'ytick.right': True,
                            'axes.spines.left': True,
                            'axes.spines.bottom': True,
                            'axes.spines.right': True,
                            'axes.spines.top': True}

            X = img[TrainingData > 0, :]
            y = TrainingData[TrainingData > 0]

            features = pd.DataFrame(X)

            band_names = []
            for i in range(X.shape[1]):
                # for i in range(0,2500):
                nband = "Band_" + str(i + 1)
                band_names.append(nband)

            # print("*******************")
            # print(band_names)
            # print("*******************")

            features.columns = band_names

            print('The shape of our features is:', features.shape)
            print('The number of Spectra is:', features.shape[0])
            print('The number of bands is:', features.shape[1])

            features['value'] = y

            # min_max_scaler = preprocessing.MinMaxScaler()
            #
            # xscaled = min_max_scaler.fit_transform(features)
            # features_ = pd.DataFrame(xscaled)
            #
            # features_.transpose().plot(figsize=(20, 7))
            # plt.legend(bbox_to_anchor=(0.1, -0.1), loc='upper left', ncol=7)
            # plt.title('Reference Spectra')
            # plt.plot()
            #
            # # # In[10]:

            # Labels are the values we want to predict
            labels = np.array(features['value'])

            # Remove the labels from the features
            # axis 1 refers to the columns
            features = features.drop('value', axis=1)

            # Saving feature names for later use
            feature_list = list(features.columns)

            # Convert to numpy array
            features = np.array(features)

            print('Training Features Shape: ', features.shape)
            print('Training Labels Shape: ', labels.shape)



            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 730, 'Section 1:')
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 710, "Image extent: " + str(img.shape[0]) +" x "+ str(img.shape[1]) + " (Rows x Columns)")
            doc.drawString(30, 690, "Number of Bands: " + str(features.shape[1]))
            doc.drawString(30, 670, "Field name (shape file) of your classes: " + attributes)
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 650, " Random Forrest Training")
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 630, " Number of Tress: " )
            doc.drawString(30, 610, " Number of samples: " + str(n_samples))
            #doc.drawString(30, 590, " Split size for Test: " + str(RFHelper.helper.SplitSize))
            #doc.drawString(30, 570, " Training samples: " + str(labels.shape[0]))
            #doc.drawString(30, 550, " Test sample: " + str(RFHelper.helper.TestSample))

            #doc.drawString(30, 530, "Co efficient of determination R^2 of the prediction : " + RFHelper.helper.Coeff)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 510, "left: ordered by band number| right: ordered by importance ")

            imp = {}
            for i in range(len(RFR.feature_importances_)):
                importance = round(RFR.feature_importances_[i] * 100, 2)
                print('Band {}: {}'.format(i + 1, importance))
                imp['Band{}'.format(i + 1)] = importance

            data = [(k, v) for k, v in imp.items()]
            data2 = data[:]

            data2.sort(key=lambda x: x[1], reverse=True)

            data.insert(0, ("Band", "Importance"))
            data2.insert(0, ("Band", "Importance"))

            doc.setFont('Helvetica', 10)

            if (len(data) > 21):
                chunks = (self.chunks(data, 20))
                ordered_chunks = (self.chunks(data2, 20))
                iterationNumer = 0
                for unorder, ordered in zip(chunks, ordered_chunks):
                    if iterationNumer != 0:
                        unorder.insert(0, ("Band", "Importance"))
                        ordered.insert(0, ("Band", "Importance"))
                    unordered_table = Table(unorder)
                    ordered_table = Table(ordered)
                    unordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    ordered_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                                       ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                    if iterationNumer == 0:
                        unordered_table.wrapOn(doc, 60, 100)
                        unordered_table.drawOn(doc, 60, 100)

                        ordered_table.wrapOn(doc, 280, 100)
                        ordered_table.drawOn(doc, 280, 100)
                    else:
                        unordered_table.wrapOn(doc, 60, 400)
                        unordered_table.drawOn(doc, 60, 400)

                        ordered_table.wrapOn(doc, 280, 400)
                        ordered_table.drawOn(doc, 280, 400)

                    columns = [""]
                    ordered_table = Table(columns)
                    unordered_table = Table(columns)
                    doc.showPage()
                    iterationNumer += 1

            else:
                table = Table(data)
                table2 = Table(data2)

                table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                           ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                                            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

                table.wrapOn(doc, 0, 0)
                table.drawOn(doc, 60, 100)

                table2.wrapOn(doc, 60, 100)
                table2.drawOn(doc, 280, 100)
                doc.showPage()

            score_c = r2_score(y, y_c)
            score_cv = r2_score(y, y_cv)
            mse_c = mean_squared_error(y, y_c)
            mse_cv = mean_squared_error(y, y_cv)

            errors = abs(y_cv - labels)
            # Print out the mean absolute error (mae)

            # Print out the mean absolute error (mae)

            print('-------------')
            print('n of the test data: {}'.format(len(labels)))
            print('Mean of the variable: {:.2f}'.format(np.mean(labels)))
            print('Standard deviation of the variable: {:.2f}'.format(np.std(labels)))
            print('-------------')
            print('Mean Absolute Error: {:.2f}'.format(np.mean(errors)))

            mse = mean_squared_error(labels, y_cv)
            print('Mean squared error: {:.2f}'.format(mse))
            print('RMSE: {:.2f}'.format(np.sqrt(mse)))
            print(
                'RPD: {:.2f} | How often does RMSE of Prediction fit in the Standard Deviation of the samples'.format(
                    np.std(labels) / np.sqrt(mse)))

            doc.drawString(30, 770, "n of the data: " + str(len(labels)))
            doc.drawString(30, 750, "Mean of the Variable : " + str(np.mean(labels)))
            doc.drawString(30, 730, "Standard Deviation of the Variable: " + str(round(np.std(labels), 2)))
            doc.drawString(30, 710, "----------------------------------------")
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 670, " Mean Absolute Error : " + str(round(np.mean(errors), 2)))
            doc.drawString(30, 650, " Mean Squared Error:" + str(round(mse, 2)))
            doc.drawString(30, 630, " RMSE:" + str(round(np.sqrt(mse), 2)))

            doc.drawString(30, 600, 'RPD:'+ str(round(np.std(labels) / np.sqrt(mse),2)))
            doc.setFont('Helvetica', 10)
            doc.drawString(30, 580,'How often does RMSE of Prediction fit in the Standard Deviation of the samples')
            doc.drawString(30, 560, "----------------------------------------")
            mape = 100 * (errors / labels)
            # Calculate and display accuracy
            accuracy = 100 - np.mean(mape)
            print('mean absolute percentage error (MAPE) / Accuracy: {:.2f}'.format(accuracy), '%.')
            print('-------------')
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 530,'mean absolute percentage error (MAPE) / Accuracy:'+ str(round(accuracy,2))+ ' %.')
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination r²: {:.2f}'.format(r2_score(labels, y_cv)))
            doc.drawString(30, 510,'Coefficient of determination r²: '+ str(round(r2_score(labels, y_cv),2)))


            '''
            To put our predictions in perspective, we can calculate an accuracy using
            the mean average percentage error subtracted from 100 %.
            '''

            # Calculate mean absolute percentage error (MAPE)



            print('R2 calib: %5.3f' % score_c)
            print('R2 LOOCV: %5.3f' % score_cv)
            print('MSE calib: %5.3f' % mse_c)
            print('MSE LOOCV: %5.3f' % mse_cv)

            # In[12]:
            #
            # Plot regression and figures of merit
            rangey = max(y) - min(y)
            rangex = max(y_c) - min(y_c)

            # Fit a line to the CV vs response
            z = np.polyfit(y, y_c, 1)
            with plt.style.context(('ggplot')):
                sns.set(rc=settings_sns, font_scale=1.0)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_c, y, c='red', edgecolors='k')
                # Plot the best fit line
                ax.plot(np.polyval(z, y), y, c='blue', linewidth=1)
                # Plot the ideal 1:1 line
                ax.plot(y, y, color='green', linewidth=1)
                plt.title(attributes)
                plt.xlabel('predicted')
                plt.ylabel('observed')
                plt.legend(['regression \nr$^{2}$ (CV): ' + str(round(score_cv, 2)), '1:1 line', 'samples'])

                imgdata = BytesIO()
                fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
                plt.savefig(path + "r2_Prediction.png", dpi=300)

                imgdata.seek(0)  # rewind the data

                drawing = svg2rlg(imgdata)
                renderPDF.draw(drawing, doc, 30, 70)
                fig.clf()
                plt.close(fig)
            #

            return doc
        except ValueError as e:
            print(e)


    def chunks(self,lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

