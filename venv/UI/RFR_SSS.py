try:
    import sys,os
    sys.path.append(r"F:\Work\Maptor\venv\Model")
    from ReportModule import ReportModule
    sys.path.append(r"F:\Work\Maptor\venv\Model")
    from InputController import InputController
    sys.path.append(r"F:\Work\Maptor\venv\HelpingModel")
    from RFHelper import RFHelper
    sys.path.append(r"..\HelpingModel")
    from RegRptHelper import RegressionReportHelper
    from RFR_SSS_Controller import RFR_SSS_Controller
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QFileDialog,QMessageBox
    from osgeo import ogr

    from osgeo import gdal, ogr, gdal_array  # I/O image data
    import numpy as np  # math and array handling
    import matplotlib.pyplot as plt  # plot figures
    import pandas as pd  # handling large data as table sheets
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


class Ui_RFR_SSS(object):

    input_C = InputController()
    RgMdl = RFR_SSS_Controller()
    rt = ReportModule()
    saveModel = 0


    def setupUi(self, RFR_SSS):
        RFR_SSS.setObjectName("RFR_SSS")
        RFR_SSS.resize(870, 845)
        self.centralwidget = QtWidgets.QWidget(RFR_SSS)
        self.centralwidget.setObjectName("centralwidget")
        self.ImgPath = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgPath.setGeometry(QtCore.QRect(10, 160, 311, 31))
        self.ImgPath.setObjectName("ImgPath")
        self.tranData_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.tranData_Path.setGeometry(QtCore.QRect(10, 240, 311, 31))
        self.tranData_Path.setObjectName("tranData_Path")
        self.ImgPth_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.ImgPth_Btn.setGeometry(QtCore.QRect(370, 160, 93, 31))
        self.ImgPth_Btn.setObjectName("ImgPth_Btn")
        self.tranPath_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.tranPath_Btn.setGeometry(QtCore.QRect(370, 240, 93, 31))
        self.tranPath_Btn.setObjectName("tranPath_Btn")
        self.attr_btn = QtWidgets.QPushButton(self.centralwidget)
        self.attr_btn.setGeometry(QtCore.QRect(740, 240, 93, 31))
        self.attr_btn.setObjectName("attr_btn")
        self.attributes = QtWidgets.QComboBox(self.centralwidget)
        self.attributes.setGeometry(QtCore.QRect(680, 190, 151, 31))
        self.attributes.setObjectName("attributes")
        self.RunBtn = QtWidgets.QPushButton(self.centralwidget)
        self.RunBtn.setGeometry(QtCore.QRect(710, 320, 121, 41))
        self.RunBtn.setObjectName("RunBtn")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 510, 861, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(60, 30, 751, 61))
        font = QtGui.QFont()
        font.setPointSize(23)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.savePath = QtWidgets.QLineEdit(self.centralwidget)
        self.savePath.setGeometry(QtCore.QRect(10, 580, 321, 31))
        self.savePath.setObjectName("savePath")
        self.PathLabel = QtWidgets.QLabel(self.centralwidget)
        self.PathLabel.setGeometry(QtCore.QRect(10, 550, 131, 20))
        self.PathLabel.setObjectName("PathLabel")
        self.BrowsePath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowsePath.setGeometry(QtCore.QRect(370, 580, 93, 31))
        self.BrowsePath.setObjectName("BrowsePath")
        self.SaveBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveBtn.setGeometry(QtCore.QRect(710, 560, 111, 21))
        self.SaveBtn.setObjectName("SaveBtn")
        self.ReportNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportNameLabel.setGeometry(QtCore.QRect(10, 640, 131, 20))
        self.ReportNameLabel.setObjectName("ReportNameLabel")
        self.ProjectName = QtWidgets.QLineEdit(self.centralwidget)
        self.ProjectName.setGeometry(QtCore.QRect(10, 670, 221, 31))
        self.ProjectName.setObjectName("ProjectName")
        self.ImgPath_label = QtWidgets.QLabel(self.centralwidget)
        self.ImgPath_label.setGeometry(QtCore.QRect(10, 130, 131, 20))
        self.ImgPath_label.setObjectName("ImgPath_label")
        self.TrainData_label = QtWidgets.QLabel(self.centralwidget)
        self.TrainData_label.setGeometry(QtCore.QRect(10, 210, 131, 20))
        self.TrainData_label.setObjectName("TrainData_label")
        self.trees_label = QtWidgets.QLabel(self.centralwidget)
        self.trees_label.setGeometry(QtCore.QRect(20, 350, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.trees_label.setFont(font)
        self.trees_label.setObjectName("trees_label")
        self.TreesNo = QtWidgets.QLineEdit(self.centralwidget)
        self.TreesNo.setGeometry(QtCore.QRect(20, 390, 113, 41))
        self.TreesNo.setObjectName("TreesNo")
        RFR_SSS.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RFR_SSS)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 870, 26))
        self.menubar.setObjectName("menubar")
        RFR_SSS.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RFR_SSS)
        self.statusbar.setObjectName("statusbar")
        RFR_SSS.setStatusBar(self.statusbar)

        self.TreesNo.setText("1000")
        self.ImgPth_Btn.clicked.connect(self.openImagePathDialog)
        self.tranPath_Btn.clicked.connect(self.openTrainingFileDialog)
        self.BrowsePath.clicked.connect(self.BrowsePathDir)
        self.SaveBtn.toggled.connect(self.SaveModel)
        self.attr_btn.clicked.connect(self.Get_TrainAttribute)
        self.RunBtn.clicked.connect(self.validateInput)

        self.retranslateUi(RFR_SSS)
        QtCore.QMetaObject.connectSlotsByName(RFR_SSS)

    def retranslateUi(self, RFR_SSS):
        _translate = QtCore.QCoreApplication.translate
        RFR_SSS.setWindowTitle(_translate("RFR_SSS", "Regression"))
        self.ImgPth_Btn.setText(_translate("RFR_SSS", "Browse"))
        self.tranPath_Btn.setText(_translate("RFR_SSS", "Browse"))
        self.attr_btn.setText(_translate("RFR_SSS", "Get attributes"))
        self.RunBtn.setText(_translate("RFR_SSS", "Run"))
        self.label.setText(_translate("RFR_SSS", "Random Forrest Regression - LOOCV"))
        self.PathLabel.setText(_translate("RFR_SSS", "Results Directory"))
        self.BrowsePath.setText(_translate("RFR_SSS", "Browse"))
        self.SaveBtn.setText(_translate("RFR_SSS", "Save Model"))
        self.ReportNameLabel.setText(_translate("RFR_SSS", "Project Name"))
        self.ImgPath_label.setText(_translate("RFR_SSS", "Image Path"))
        self.TrainData_label.setText(_translate("RFR_SSS", "Training Data Path"))
        self.trees_label.setText(_translate("RFR_SSS", "Number of Trees"))



    def openImagePathDialog(self):
        filename = QFileDialog.getOpenFileName()
        self.ImgPath.setText(filename[0])
        self.input_C.set_img_path(filename[0])

    def openTrainingFileDialog(self):
        filename = QFileDialog.getOpenFileName()
        self.tranData_Path.setText(filename[0])
        self.input_C.set_training_path(filename[0])

    def Get_TrainAttribute(self):
        if not self.tranData_Path.text().endswith(".shp"):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Wrong Selection")
            msg.setText("Please select correct file with .shp extension to get attributes. ")
            msg.exec_()
            return
        else:
            attlist = self.FindAttributes(self.tranData_Path.text())
            if(attlist):
                self.attributes.addItems(attlist)
            if not (attlist):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("NO ATTRIBUTES FOUND")
                msg.setText("NO ATTRIBUTES FOUND IN .SHP FILE. ")
                msg.exec_()

    def FindAttributes(self, filepath):
        return self.input_C.FindAttributes(filepath)
        # try:
        #     driver = ogr.GetDriverByName('ESRI Shapefile')
        #     shape_dataset = driver.Open(filepath)
        #     shape_layer = shape_dataset.GetLayer()
        #     field_names = [field.name for field in shape_layer.schema]
        #     return field_names
        # except ValueError as e:
        #     print(e)

    def BrowsePathDir(self):
        filename = QFileDialog.getExistingDirectory()
        self.savePath.setText(filename)

    def BrowseImagePath(self):
        filename = QFileDialog.getExistingDirectory()
        self.ImgSavePath.setText(filename)

    def SaveModel(self):
        if self.SaveBtn.isChecked():
            self.saveModel = 1
            # self.ModelNameLabel.show()
            # self.ModelPathLabel.show()
            # self.BrowseMdlPath.show()
            # self.ModelSavePath.show()
            # self.ModelName.show()
        else:
            self.saveModel = 0
            # self.ModelNameLabel.hide()
            # self.ModelPathLabel.hide()
            # self.ModelSavePath.hide()
            # self.BrowseMdlPath.hide()
            # self.ModelName.hide()

    def BrowseModelPath(self):
        filename = QFileDialog.getExistingDirectory()
        self.ModelSavePath.setText(filename)

    def validateInput(self):

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Validation Prompt")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if not self.ImgPath.text().endswith(".tif"):
            msg.setText("Invalid Image File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText("The details are as follows: Either Image file is not selected or does not contain .tif extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RFR_SSS()
            self.ui.setupUi(self.window)
            return

        if not self.tranData_Path.text().endswith(".shp"):
            msg.setText("Invalid Training Data File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText("The details are as follows: Either Training file is not selected or does not contain .shp extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RFR_SSS()
            self.ui.setupUi(self.window)
            return

        if self.attributes.currentText() == "" :
            msg.setText("Please select the attribute. ")
            msg.setInformativeText("Attribute not selected")
            msg.setDetailedText("The details are as follows: Select attribute from Training File")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RFR_SSS()
            self.ui.setupUi(self.window)
            return


        if self.ProjectName.text() == "":
            msg.setText("Please enter a suitable name for your project. ")
            msg.setInformativeText("project name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RFR_SSS()
            self.ui.setupUi(self.window)
            return

        if self.savePath.text() == "":
            msg.setText("Please select path for project. ")
            msg.setInformativeText("project path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RFR_SSS()
            self.ui.setupUi(self.window)
            return

        # if self.TreesNo.text() == "":
        #     msg.setText("Please enter a suitable numbe. ")
        #     msg.setInformativeText("Image name missing")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_RFR_SSS()
        #     self.ui.setupUi(self.window)
        #     return
        #
        # if self.ImgSavePath.text() == "":
        #     msg.setText("Please select path for Image result. ")
        #     msg.setInformativeText("Image path missing")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_RFR_SSS()
        #     self.ui.setupUi(self.window)
        #     return
        #
        # if self.saveModel == 1 and (self.ModelName.text() == "" or self.ModelSavePath.text() == ""):
        #     msg.setText("Please enter name and Path for saving model. ")
        #     msg.setInformativeText("Either Model Name of Path not entered")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_RFR_SSS()
        #     self.ui.setupUi(self.window)
        #     return

        else:
            self.Run()

    def Run(self):

        try:
            print("The process has started.... This will take few minutes")
            if self.TreesNo.text() == "":
                self.TreesNo.setText("1000")


            treeNo = int(self.TreesNo.text())

            print(self.savePath.text())
            dir_path = self.savePath.text() + "\\" + self.ProjectName.text()

            if os.path.isdir(dir_path):

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Validation Prompt")
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

                msg.setText("A folder nammed " + self.ProjectName.text() + " already exists. ")
                msg.setInformativeText("Please move the folder and try again")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_RFR_SSS()
                self.ui.setupUi(self.window)
                return

            else:
                os.mkdir(dir_path)

                reportpath = dir_path
                reportpath += "/" + str(self.ProjectName.text()) + "_REPORT.pdf"

                prediction_map = dir_path
                prediction_map += "/" + str(self.ProjectName.text()) + "_IMG.tif"
                modelsavepath = dir_path + "/" + self.ProjectName.text() + "_MODEL.sav"

                LoadingImages = self.input_C.load_image_data()
                img_ds = LoadingImages[0]
                img = LoadingImages[1]

                doc = self.rt.build_doc(reportpath, "Random Forrest Regression Sparse Report")
                self.input_C.set_training_attr(self.attributes.currentText())

                TrainingData = self.input_C.load_train_data(img_ds, "Regression")  ## roi
                print(TrainingData)

                # plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
                # roi_positions = np.where(TrainingData > 0)
                # plt.scatter(roi_positions[1], roi_positions[0], marker='x', c='r')
                # plt.title('first RS band and sample points')
                #
                # plt.show()

                # In[7]:

                # Number of training pixels:
                n_samples = (TrainingData > 0).sum()
                print('We have {n} training samples'.format(
                    n=n_samples))  # Subset the image dataset with the training image = X

                # In[8]:

                # Mask the classes on the training dataset = y
                # These will have n_samples rows
                # self.RgMdl.RFR_Regressor(TrainingData,img,self.attributes.currentText(),test_size,est)

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

                # In[11]:
                randomState = 35
                y_c, y_cv,RFR = self.RgMdl.RFR_Regressor(X,y,treeNo,features,labels)
                new_shape = (img.shape[0] * img.shape[1], img.shape[2])
                img_as_array = img[:, :, :].reshape(new_shape)

                print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

                img_as_array = np.nan_to_num(img_as_array)

                # In[17]:

                prediction_ = self.RgMdl.RFR_prediction(RFR,img_as_array)

                # prediction_ = RFR.predict(img_as_array)

                # In[18]:

                prediction = prediction_.reshape(img[:, :, 0].shape)
                print('Reshaped back to {}'.format(prediction.shape))


                self.rt.make_rfr_sss_report(doc,reportpath,dir_path,self.tranData_Path.text(),prediction_map,modelsavepath,self.ImgPath.text(),TrainingData,img,self.attributes.currentText(),prediction,y_c, y_cv, RFR)
                doc.save()

                self.RgMdl.RF_save_PredictionImage(prediction_,prediction_map,img,img_ds)

                if self.saveModel == 1:
                    self.RgMdl.save_model(RFR, modelsavepath)

                print('report is ready')
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Complete!!!")
                msg.setText("Processing Done. Report ready to preview ")
                msg.exec_()

        except ValueError as e:
            print(e)
