from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
from reportlab.platypus import Table,TableStyle,Frame,KeepInFrame
from reportlab.lib.units import inch
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
import seaborn as sn
import pandas as pd
import operator
import pprint


class ReportModule():
  
    def build_doc(self,path,mode):
        try:
            doc = canvas.Canvas(path)
            doc.setLineWidth(.3)
            doc.setFont('Helvetica-Bold', 16)
            doc.drawString(30, 810, "Random Forrest "+ mode + " Report")
            doc.setFont('Helvetica', 12)
            doc.drawString(30, 793, 'Software Maptor v2.0')
            doc.drawString(30, 779, "Developer: Florian Beyer & Qasim Usmani")
            doc.drawString(30, 765, 'Support: florian.beyer@uni-rostock.de')
            today = date.today()
            doc.drawString(30, 751, "Date: " + str(today))
            doc.drawString(30, 737, '')
            doc.line(20, 735, 570, 735)
            return doc
        except Error as e:
            print(e)

    def prepare_trg_subplots(self,doc,data1,data2):

        # fig = plt.figure(figsize=(6 ,6))
        # fig.suptitle('Training data', fontsize=14)
        #
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplot(121)
        # plt.imshow(data1, cmap=plt.cm.Greys_r)  # data  = img[:, :, 0] &&& cmap = plt.cm.Greys_r
        # plt.title('RS image - first band')
        # plt.subplot(122)
        # plt.imshow(data2, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
        # plt.title('Training Image')
        # plt.show()
        # imgdata = BytesIO()
        # fig.savefig(imgdata, format='svg',bbox_inches='tight',pad_inches=0)
        # imgdata.seek(0)  # rewind the data
        # drawing = svg2rlg(imgdata)
        # renderPDF.draw(drawing, doc, 50, 400)
        # doc.line(20, 350, 570, 350)
        # doc.setLineWidth(.3)
        # doc.setFont('Helvetica-Bold', 14)
        # doc.drawString(30, 620, 'Section 1: General Information and Training')
        # doc.drawString(30, 600, "Section 2: Validation")
        # doc.line(20, 650, 570, 650)
        return

    def Clf_prepare_report(self,doc,img,roi,importance,table_M,trees,ob_score,class_prediction,ValidationData,attribute):
        try:
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
            doc.showPage()



            doc.line(20, 810, 570, 810)
            doc.setLineWidth(.3)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 790, 'Section 1: General Information and Training')
            doc.drawString(30, 770, "Section 2: Validation")
            doc.line(20, 750, 570, 750)

            n_samples = (roi > 0).sum()

            doc.drawString(30, 730, "Section :1")

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
                ordered_chunks = (self.chunks(data, 20))
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
        except Error as e:
            print(e)

    # def prepare_class_prediction(self,doc,img,class_prediction):
    #     mask = np.copy(img[:, :, 0])
    #     mask[mask > 0.0] = 1.0  # all actual pixels have a value of 1.0
    #
    #     fig = plt.figure(figsize=(7,6))
    #     plt.imshow(mask)
    #
    #     class_prediction.astype(np.float16)
    #     class_prediction_ = class_prediction * mask
    #     #
    #     plt.subplot(121)
    #     plt.imshow(class_prediction, cmap=plt.cm.Spectral)
    #     plt.title('classification unmasked')
    #
    #     plt.subplot(122)
    #     plt.imshow(class_prediction_, cmap=plt.cm.Spectral)
    #     plt.title('classification masked')
    #
    #     imgdata = BytesIO()
    #     fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)
    #
    #     imgdata.seek(0)  # rewind the data
    #
    #     drawing = svg2rlg(imgdata)
    #     renderPDF.draw(drawing, doc, 30, 590)
    #     fig.clf()
    #     return doc

    def Clf_prepare_section3(self,doc,X_v,y_v,convolution_mat,df_sum_mat,score,roi_v,model_path,Image_savePath):
        try:

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

            doc.drawString(30, 260,"Precision, recall, F-measure and support for each class:")

            t1.wrapOn(doc, 30, 100)
            t1.drawOn(doc, 90, 70)

            # acc_col_name = ['accuracy','macro avg','weighted avg']
            # accuracy.insert(0,".",acc_col_name)
            # acc_data = np.array(accuracy).tolist()
            # t2 = Table(acc_data)
            #
            # t2.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            #                         ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))
            # t2.wrapOn(doc, 30, 100)
            # t2.drawOn(doc, 330, 100)
            doc.setFont('Helvetica', 11)
            doc.drawString(30, 40, "Image Saved to Path: "+Image_savePath)
            if model_path!="/":
                doc.drawString(30, 20, "Model Saved to Path: Model not Save")
            else:
                doc.drawString(30, 20, "Model Saved to Path:"+ model_path)

            return doc
        except Error as e:
            print(e)

    def Reg_prepare_report(self,doc,img,prediction,training,helper,img_path,train_data,reportpath,prediction_map,modelsavepath):
        # Display images
        try:
            width, height = A4
            print(reportpath)
            print(prediction_map)
            print(modelsavepath)
            print(img_path)
            print(train_data)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 700,"DIRECTORIES:")

            doc.setFont('Helvetica', 14)
            doc.drawString(30, 680, "Remote Sensing Image: " + img_path)
            doc.drawString(30, 660, "Shape file: " + train_data)
            doc.drawString(30, 640, "Report Save Path: "+ reportpath)
            doc.drawString(30, 620, "Regression image saved to: " +prediction_map)
            if modelsavepath !='/':
                doc.drawString(30, 600, "Model saved to: "+modelsavepath)
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
                plt.imshow(training, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
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
                plt.imshow(training, cmap=plt.cm.Spectral)  # data = roi && cmap = plt.cm.Spectral
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


            doc.line(20,800, 570, 800)
            doc.setLineWidth(.3)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 780, 'Section 1: General Information and Training')
            doc.drawString(30, 765, "Section 2: Validation")
            doc.line(20, 750, 570, 750)



            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 730, 'Section 1:')
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 710,"Image extent: "+helper.imageExtent + "(Rows x Columns)")
            doc.drawString(30, 690, "Number of Bands: "+ str(helper.BandNumber))
            doc.drawString(30, 670, "Field name (shape file) of your classes: "+helper.FieldName)
            doc.setFont('Helvetica', 14)
            doc.drawString(30,650," Random Forrest Training")
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 630, " Number of Tress: "+ str(helper.TreesNo))
            doc.drawString(30, 610, " Number of samples: "+ str(helper.SampleNo))
            doc.drawString(30, 590, " Split size for Test: "+str(helper.SplitSize))
            doc.drawString(30, 570, " Training samples: "+str(helper.TrainingSample))
            doc.drawString(30, 550, " Test sample: "+str(helper.TestSample))

            doc.drawString(30, 530, "Co efficient of determination R^2 of the prediction : "+helper.Coeff)
            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 510, "left: ordered by band number| right: ordered by importance ")

            data = ["Band", "Importance"]
            data = [(k, v) for k, v in helper.BandImportance.items()]

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


            #data.extend(dummy)
            data2.sort(key=lambda x: x[1], reverse=True)

            data.insert(0, ("Band", "Importance"))
            data2.insert(0, ("Band", "Importance"))

            datalen = len(data)
            print((datalen))

            if(len(data)>21):
                chunks = (self.chunks(data,20))
                ordered_chunks = (self.chunks(data, 20))
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

            doc.drawString(30, 770, "n of the data: " + str(helper.n_testdata))
            doc.drawString(30, 750, "Mean of the Variable : " + str(helper.meanvariable))
            doc.drawString(30, 730, "Standard Deviation of the Variable: " + str(round(helper.stdDev,2)))
            doc.drawString(30, 710, "----------------------------------------")
            doc.setFont('Helvetica', 14)
            doc.drawString(30, 670, " Mean Absolute Error : " + str(round(helper.absError,2)))
            doc.drawString(30, 650, " Mean Squared Error:" + str(round(helper.meanSqError,2)))
            doc.drawString(30, 630, " RMSE:" + str(round(helper.RMSE,2)))
            x = str(round(helper.MAPE,2))

            doc.drawString(30, 600, " Mean absoulute percentage error(MAPE)/ Accuracy :"  + x + " %")
            doc.drawString(30, 560, "----------------------------------------")

            doc.setFont('Helvetica-Bold', 14)
            doc.drawString(30, 540, "Co efficient of determination R^2 of the prediction : " + str(round(helper.coeffR,2)))

            fig, ax = plt.subplots()
            ax.scatter(helper.testlabels, helper.pred_test_ds)
            ax.plot([helper.testlabels.min(), helper.testlabels.max()], [helper.testlabels.min(), helper.testlabels.max()], 'k--', lw=1)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')

            imgdata = BytesIO()
            fig.savefig(imgdata, format='svg', bbox_inches='tight', pad_inches=0)

            imgdata.seek(0)  # rewind the data

            drawing = svg2rlg(imgdata)
            renderPDF.draw(drawing, doc, 30, 110)
            fig.clf()
            plt.close(fig)

            return doc
        except Error as e:
            print(e)

    def chunks(self,lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

