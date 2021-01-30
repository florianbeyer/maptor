try:
    import sys,os,logging
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    import matplotlib.pyplot as plt # plot figures
    import numpy as np
    sys.path.append(r"F:\Work\Maptor\venv\Controller")
    from InputController import InputController
    from PLSR_LDS_Controller import PLSR_LDS_Controller
    sys.path.append(r"F:\Work\Maptor\venv\Model")
    from ReportModule import ReportModule
    sys.path.append(r"F:\Work\Maptor\venv\HelpingModel")
    from PLSR_LDS_Helper import PLSR_LDS_Helper
    import pandas as pd
    import traceback

    from osgeo import gdal, ogr, gdal_array  # I/O image data
    import numpy as np  # math and array handling
    import matplotlib.pyplot as plt  # plot figures
    import pandas as pd  # handling large data as table sheets
    from joblib import dump, load
    from operator import itemgetter

    from sklearn.model_selection import train_test_split
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict

except Exception as e:
    logging.error("Exception occurred", exc_info=True)
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)



class Ui_PLSR_LDS(object):
    input_C = InputController()
    PLSR_C = PLSR_LDS_Controller()
    rt = ReportModule()


    def setupUi(self, PLSR_LDS):
        PLSR_LDS.setObjectName("PLSR_LDS")
        PLSR_LDS.resize(881, 844)
        self.centralwidget = QtWidgets.QWidget(PLSR_LDS)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowsePath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowsePath.setGeometry(QtCore.QRect(390, 620, 93, 31))
        self.BrowsePath.setObjectName("BrowsePath")
        self.TrainData_label = QtWidgets.QLabel(self.centralwidget)
        self.TrainData_label.setGeometry(QtCore.QRect(30, 220, 131, 20))
        self.TrainData_label.setObjectName("TrainData_label")
        self.ImgPth_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.ImgPth_Btn.setGeometry(QtCore.QRect(390, 170, 93, 31))
        self.ImgPth_Btn.setObjectName("ImgPth_Btn")
        self.savePath = QtWidgets.QLineEdit(self.centralwidget)
        self.savePath.setGeometry(QtCore.QRect(10, 620, 321, 31))
        self.savePath.setObjectName("savePath")
        self.PathLabel = QtWidgets.QLabel(self.centralwidget)
        self.PathLabel.setGeometry(QtCore.QRect(10, 590, 131, 20))
        self.PathLabel.setObjectName("PathLabel")
        self.attr_btn = QtWidgets.QPushButton(self.centralwidget)
        self.attr_btn.setGeometry(QtCore.QRect(700, 200, 93, 31))
        self.attr_btn.setObjectName("attr_btn")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 510, 861, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(160, 30, 671, 61))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.ImgPath_label = QtWidgets.QLabel(self.centralwidget)
        self.ImgPath_label.setGeometry(QtCore.QRect(30, 140, 131, 20))
        self.ImgPath_label.setObjectName("ImgPath_label")
        self.SaveBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveBtn.setGeometry(QtCore.QRect(730, 600, 111, 21))
        self.SaveBtn.setObjectName("SaveBtn")
        self.tranPath_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.tranPath_Btn.setGeometry(QtCore.QRect(390, 250, 93, 31))
        self.tranPath_Btn.setObjectName("tranPath_Btn")
        self.ReportNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportNameLabel.setGeometry(QtCore.QRect(10, 680, 131, 20))
        self.ReportNameLabel.setObjectName("ReportNameLabel")
        self.ImgPath = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgPath.setGeometry(QtCore.QRect(30, 170, 311, 31))
        self.ImgPath.setObjectName("ImgPath")
        self.ProjectName = QtWidgets.QLineEdit(self.centralwidget)
        self.ProjectName.setGeometry(QtCore.QRect(10, 710, 221, 31))
        self.ProjectName.setObjectName("ProjectName")
        self.RunBtn = QtWidgets.QPushButton(self.centralwidget)
        self.RunBtn.setGeometry(QtCore.QRect(670, 280, 121, 41))
        self.RunBtn.setObjectName("RunBtn")
        self.tranData_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.tranData_Path.setGeometry(QtCore.QRect(30, 250, 311, 31))
        self.tranData_Path.setObjectName("tranData_Path")
        self.attributes = QtWidgets.QComboBox(self.centralwidget)
        self.attributes.setGeometry(QtCore.QRect(640, 150, 151, 31))
        self.attributes.setObjectName("attributes")
        self.testsize = QtWidgets.QLineEdit(self.centralwidget)
        self.testsize.setGeometry(QtCore.QRect(40, 400, 111, 41))
        self.testsize.setObjectName("testsize")
        self.SplitSize_label = QtWidgets.QLabel(self.centralwidget)
        self.SplitSize_label.setGeometry(QtCore.QRect(30, 340, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.SplitSize_label.setFont(font)
        self.SplitSize_label.setObjectName("SplitSize_label")
        self.trnsplt_label = QtWidgets.QLabel(self.centralwidget)
        self.trnsplt_label.setGeometry(QtCore.QRect(160, 400, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.trnsplt_label.setFont(font)
        self.trnsplt_label.setObjectName("trnsplt_label")
        self.SplitSize_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.SplitSize_label_2.setGeometry(QtCore.QRect(30, 460, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.SplitSize_label_2.setFont(font)
        self.SplitSize_label_2.setObjectName("SplitSize_label_2")
        PLSR_LDS.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PLSR_LDS)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 881, 26))
        self.menubar.setObjectName("menubar")
        PLSR_LDS.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(PLSR_LDS)
        self.statusbar.setObjectName("statusbar")
        PLSR_LDS.setStatusBar(self.statusbar)
        self.testsize.setText("25")
        self.ImgPth_Btn.clicked.connect(self.openImagePathDialog)
        self.attr_btn.clicked.connect(self.Get_TrainAttribute)
        self.tranPath_Btn.clicked.connect(self.openTrainingFileDialog)
        self.BrowsePath.clicked.connect(self.BrowsePathDir)
        self.RunBtn.clicked.connect(self.validateInput)

        self.retranslateUi(PLSR_LDS)
        QtCore.QMetaObject.connectSlotsByName(PLSR_LDS)

    def retranslateUi(self, PLSR_LDS):
        _translate = QtCore.QCoreApplication.translate
        PLSR_LDS.setWindowTitle(_translate("PLSR_LDS", "PLSR_Window"))
        self.BrowsePath.setText(_translate("PLSR_LDS", "Browse"))
        self.TrainData_label.setText(_translate("PLSR_LDS", "Training Data Path"))
        self.ImgPth_Btn.setText(_translate("PLSR_LDS", "Browse"))
        self.PathLabel.setText(_translate("PLSR_LDS", "Results Directory"))
        self.attr_btn.setText(_translate("PLSR_LDS", "Get attributes"))
        self.label.setText(_translate("PLSR_LDS", "PLSR Regression - Large Sampling"))
        self.ImgPath_label.setText(_translate("PLSR_LDS", "Image Path"))
        self.SaveBtn.setText(_translate("PLSR_LDS", "Save Model"))
        self.tranPath_Btn.setText(_translate("PLSR_LDS", "Browse"))
        self.ReportNameLabel.setText(_translate("PLSR_LDS", "Project Name"))
        self.RunBtn.setText(_translate("PLSR_LDS", "Run"))
        self.SplitSize_label.setText(_translate("PLSR_LDS", "Split size of Training & Testing sets"))
        self.trnsplt_label.setText(_translate("PLSR_LDS", "% Test Size"))
        self.SplitSize_label_2.setText(_translate("PLSR_LDS", "Training Size is 100%  -  Test Size"))

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
            if (attlist):
                self.attributes.addItems(attlist)
            if not (attlist):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("NO ATTRIBUTES FOUND")
                msg.setText("NO ATTRIBUTES FOUND IN .SHP FILE. ")
                msg.exec_()

    def FindAttributes(self, filepath):
        return self.input_C.FindAttributes(filepath)

    def BrowsePathDir(self):
        filename = QFileDialog.getExistingDirectory()
        self.savePath.setText(filename)

    def validateInput(self):

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Validation Prompt")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if not self.ImgPath.text().endswith(".tif"):
            msg.setText("Invalid Image File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText(
                "The details are as follows: Either Image file is not selected or does not contain .tif extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            return

        if not self.tranData_Path.text().endswith(".shp"):
            msg.setText("Invalid Training Data File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText(
                "The details are as follows: Either Training file is not selected or does not contain .shp extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            return

        if self.attributes.currentText() == "":
            msg.setText("Please select the attribute. ")
            msg.setInformativeText("Attribute not selected")
            msg.setDetailedText("The details are as follows: Select attribute from Training File")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            return

        if self.ProjectName.text() == "":
            msg.setText("Please enter a suitable name for your project. ")
            msg.setInformativeText("project name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            return

        if self.savePath.text() == "":
            msg.setText("Please select path for project. ")
            msg.setInformativeText("project path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            return

        else:
            self.Run()

    def Run(self):

        print("The process has started.... This will take few minutes")
        print(self.savePath.text())
        dir_path = self.savePath.text() + "\\" + self.ProjectName.text()

        if os.path.isdir(dir_path):

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Validation Prompt")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            msg.setText("A folder named " + self.ProjectName.text() + " already exists. ")
            msg.setInformativeText("Please move the folder and try again")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            return

        else:
            os.mkdir(dir_path)

        if (self.testsize.text()) == "" :
            testsize = 0.25
        else:
            testsize = int(self.testsize.text())/100


        print(testsize)

        reportpath = dir_path
        reportpath += "/" + str(self.ProjectName.text()) + "_REPORT.pdf"

        prediction_map = dir_path
        prediction_map += "/" + str(self.ProjectName.text()) + "_IMG.tif"
        if self.SaveBtn.isChecked() == True:
            modelsavepath = dir_path + "/" + self.ProjectName.text() + "_MODEL.sav"
        else:
            modelsavepath = "Model not saved"

        try:

            doc = self.rt.build_doc(reportpath, "PLSR Large Dataset")

        except Exception as e:
            print(e)
            logging.debug("exception occurred " + e, exc_info=True)

        logging.debug("exception occurred", exc_info=True)

        loading_images = self.input_C.load_image_data()

        img_ds = loading_images[0]
        img = loading_images[1]

        print(self.attributes.currentText())
        self.input_C.set_training_attr(self.attributes.currentText())
        logging.debug("exception occurred", exc_info=True)

        # print(self.attributes.currentText() +" xxx OK")
        train_data = self.input_C.load_train_data(img_ds, "Regression")  ## roi

        X = img[train_data > 0, :]
        y = train_data[train_data > 0]


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
                                                                                    test_size=testsize, random_state=35)

        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)

        mse, component = self.PLSR_C.ComponentRegressor(features, X, y)

        # # Calculate and print the position of minimum in MSE
        msemin = np.argmin(mse)
        suggested_comp = msemin + 1
        print("Suggested number of components: ", suggested_comp)


        plsr = self.PLSR_C.Regressor(suggested_comp, train_features, train_labels)
        print(plsr.score(train_features, train_labels))

        importance = self.PLSR_C.vip(plsr)

        predictions_test_ds = self.PLSR_C.testpredict(plsr,test_features)


        '''
        To put our predictions in perspective, we can calculate an accuracy using
        the mean average percentage error subtracted from 100 %.
        '''
        plsr = self.PLSR_C.finaltraining(plsr,features,labels)

        print(plsr.score(train_features, train_labels))

        new_shape = (img.shape[0] * img.shape[1], img.shape[2])
        img_as_array = img[:, :, :].reshape(new_shape)

        print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

        img_as_array = np.nan_to_num(img_as_array)
        prediction_ = self.PLSR_C.finalprediction(plsr,img_as_array)
        print(prediction_)
        prediction = prediction_.reshape(img[:, :, 0].shape)
        print('Reshaped back to {}'.format(prediction.shape))

        self.PLSR_C.saveimage(img,prediction,prediction_map,img_ds)


        rp_param = PLSR_LDS_Helper( img, train_data, features, train_features, test_features, train_labels, test_labels, mse, component, predictions_test_ds, labels,
        prediction, importance, X, y, self.attributes.currentText(), reportpath, prediction_map, modelsavepath,self.ImgPath.text(),self.tranData_Path.text())


        # rp_param = PLSR_LDS_Helper(train_data,X,y,,prediction_,dir_path,reportpath,prediction_map,modelsavepath,self.ImgPath.text(),self.tranData_Path.text())

        doc = self.rt.make_plsr_lds_report(doc,dir_path,rp_param)


        if self.SaveBtn.isChecked() == True:
            self.PLSR_C.savemodel(plsr, modelsavepath)
            logging.error("Exception occurred", exc_info=True)


        logging.error("Exception occurred", exc_info=True)
        doc.save()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Complete!!!")
        msg.setText("Processing Done. Report ready to preview ")
        msg.exec_()
