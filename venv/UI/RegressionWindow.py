# WARNING! All changes made in this file will be lost!

import sys
import os
sys.path.append(r"F:\Work\Maptor\maptor\venv\Model")
from ReportModule import ReportModule
sys.path.append(r"F:\Work\Maptor\maptor\venv\Model")
from InputController import InputController
sys.path.append(r"F:\Work\Maptor\maptor\venv\HelpingModel")
from RegRptHelper import RegressionReportHelper
from RegressionController import RegressionController
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QInputDialog, QLineEdit, QFileDialog,QDialog,QProgressBar,QMessageBox
from osgeo import ogr
from osgeo import gdal, ogr, gdal_array


class Ui_RegressionWindow(object):

    input_C = InputController()
    RgMdl = RegressionController()
    rt = ReportModule()
    helper = RegressionReportHelper()
    saveModel = 0

    def setupUi(self, Regression):
        Regression.setObjectName("Regression")
        Regression.resize(681, 619)
        self.centralwidget = QtWidgets.QWidget(Regression)
        self.centralwidget.setObjectName("centralwidget")
        self.ImgPath = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgPath.setGeometry(QtCore.QRect(10, 100, 311, 31))
        self.ImgPath.setObjectName("ImgPath")
        self.tranData_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.tranData_Path.setGeometry(QtCore.QRect(10, 180, 311, 31))
        self.tranData_Path.setObjectName("tranData_Path")
        self.ImgPth_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.ImgPth_Btn.setGeometry(QtCore.QRect(340, 100, 93, 31))
        self.ImgPth_Btn.setObjectName("ImgPth_Btn")
        self.tranPath_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.tranPath_Btn.setGeometry(QtCore.QRect(340, 180, 93, 31))
        self.tranPath_Btn.setObjectName("tranPath_Btn")
        self.attr_btn = QtWidgets.QPushButton(self.centralwidget)
        self.attr_btn.setGeometry(QtCore.QRect(570, 150, 93, 31))
        self.attr_btn.setObjectName("attr_btn")
        self.attributes = QtWidgets.QComboBox(self.centralwidget)
        self.attributes.setGeometry(QtCore.QRect(510, 100, 151, 31))
        self.attributes.setObjectName("attributes")
        self.RunBtn = QtWidgets.QPushButton(self.centralwidget)
        self.RunBtn.setGeometry(QtCore.QRect(540, 220, 121, 41))
        self.RunBtn.setObjectName("RunBtn")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 280, 681, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(270, 10, 151, 41))
        font = QtGui.QFont()
        font.setPointSize(23)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.ReportsavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.ReportsavePath.setGeometry(QtCore.QRect(10, 380, 221, 31))
        self.ReportsavePath.setObjectName("ReportsavePath")
        self.ReportPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportPathLabel.setGeometry(QtCore.QRect(10, 360, 131, 20))
        self.ReportPathLabel.setObjectName("ReportPathLabel")
        self.BrowseRptPath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseRptPath.setGeometry(QtCore.QRect(240, 380, 93, 31))
        self.BrowseRptPath.setObjectName("BrowseRptPath")
        self.BrowseMdlPath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseMdlPath.setGeometry(QtCore.QRect(440, 470, 93, 31))
        self.BrowseMdlPath.setObjectName("BrowseMdlPath")
        self.ImgSavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgSavePath.setGeometry(QtCore.QRect(10, 520, 221, 31))
        self.ImgSavePath.setObjectName("ImgSavePath")
        self.ImgPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgPathLabel.setGeometry(QtCore.QRect(10, 500, 131, 20))
        self.ImgPathLabel.setObjectName("ImgPathLabel")
        self.ImgNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgNameLabel.setGeometry(QtCore.QRect(10, 430, 131, 20))
        self.ImgNameLabel.setObjectName("ImgNameLabel")
        self.ModelNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ModelNameLabel.setGeometry(QtCore.QRect(440, 350, 211, 20))
        self.ModelNameLabel.setObjectName("ModelNameLabel")
        self.SaveBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveBtn.setGeometry(QtCore.QRect(440, 300, 111, 21))
        self.SaveBtn.setObjectName("SaveBtn")
        self.ReportNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportNameLabel.setGeometry(QtCore.QRect(10, 290, 131, 20))
        self.ReportNameLabel.setObjectName("ReportNameLabel")
        self.BrowseImgSavetBtn = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImgSavetBtn.setGeometry(QtCore.QRect(240, 520, 93, 31))
        self.BrowseImgSavetBtn.setObjectName("BrowseImgSavetBtn")
        self.ImgName = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgName.setGeometry(QtCore.QRect(10, 450, 221, 31))
        self.ImgName.setObjectName("ImgName")
        self.ModelName = QtWidgets.QLineEdit(self.centralwidget)
        self.ModelName.setGeometry(QtCore.QRect(440, 370, 221, 31))
        self.ModelName.setObjectName("ModelName")
        self.ReportName = QtWidgets.QLineEdit(self.centralwidget)
        self.ReportName.setGeometry(QtCore.QRect(10, 310, 221, 31))
        self.ReportName.setObjectName("ReportName")
        self.ModelSavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.ModelSavePath.setGeometry(QtCore.QRect(440, 430, 221, 31))
        self.ModelSavePath.setObjectName("ModelSavePath")
        self.ModelPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ModelPathLabel.setGeometry(QtCore.QRect(440, 410, 131, 20))
        self.ModelPathLabel.setObjectName("ModelPathLabel")
        self.ReportPathLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.ReportPathLabel_2.setGeometry(QtCore.QRect(10, 80, 131, 20))
        self.ReportPathLabel_2.setObjectName("ReportPathLabel_2")
        self.ReportPathLabel_3 = QtWidgets.QLabel(self.centralwidget)
        self.ReportPathLabel_3.setGeometry(QtCore.QRect(10, 160, 131, 20))
        self.ReportPathLabel_3.setObjectName("ReportPathLabel_3")
        Regression.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Regression)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 681, 26))
        self.menubar.setObjectName("menubar")
        Regression.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Regression)
        self.statusbar.setObjectName("statusbar")
        Regression.setStatusBar(self.statusbar)


        self.ModelNameLabel.hide()
        self.ModelPathLabel.hide()
        self.BrowseMdlPath.hide()
        self.ModelName.hide()
        self.ModelSavePath.hide()

        self.ImgPth_Btn.clicked.connect(self.openImagePathDialog)
        self.tranPath_Btn.clicked.connect(self.openTrainingFileDialog)

        self.BrowseRptPath.clicked.connect(self.BrowseReportPath)
        self.BrowseImgSavetBtn.clicked.connect(self.BrowseImagePath)

        self.SaveBtn.toggled.connect(self.SaveModel)
        self.BrowseMdlPath.clicked.connect(self.BrowseModelPath)

        self.attr_btn.clicked.connect(self.Get_TrainAttribute)
        self.RunBtn.clicked.connect(self.validateInput)


        self.retranslateUi(Regression)
        QtCore.QMetaObject.connectSlotsByName(Regression)

    def retranslateUi(self, Regression):
        _translate = QtCore.QCoreApplication.translate
        Regression.setWindowTitle(_translate("Regression", "Regression"))
        self.ImgPth_Btn.setText(_translate("Regression", "Browse"))
        self.tranPath_Btn.setText(_translate("Regression", "Browse"))
        self.attr_btn.setText(_translate("Regression", "Get attributes"))
        self.RunBtn.setText(_translate("Regression", "Run"))
        self.label.setText(_translate("Regression", "Regression"))
        self.ReportPathLabel.setText(_translate("Regression", "Save Report to Path"))
        self.BrowseRptPath.setText(_translate("Regression", "Browse"))
        self.BrowseMdlPath.setText(_translate("Regression", "Browse"))
        self.ImgPathLabel.setText(_translate("Regression", "Result Image Path"))
        self.ImgNameLabel.setText(_translate("Regression", "Save Result Image Name"))
        self.ModelNameLabel.setText(_translate("Regression", "Model Name"))
        self.SaveBtn.setText(_translate("Regression", "Save Model"))
        self.ReportNameLabel.setText(_translate("Regression", "Report Name"))
        self.BrowseImgSavetBtn.setText(_translate("Regression", "Browse"))
        self.ModelPathLabel.setText(_translate("Regression", "Save Model to Path"))
        self.ReportPathLabel_2.setText(_translate("Regression", "Image Path"))
        self.ReportPathLabel_3.setText(_translate("Regression", "Training Data Path"))

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
            self.attributes.addItems(attlist)

    def FindAttributes(self, filepath):
        try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            shape_dataset = driver.Open(filepath)
            shape_layer = shape_dataset.GetLayer()
            field_names = [field.name for field in shape_layer.schema]
            return field_names
        except ValueError as e:
            print(e)

    def BrowseReportPath(self):
        filename = QFileDialog.getExistingDirectory()
        self.ReportsavePath.setText(filename)

    def BrowseImagePath(self):
        filename = QFileDialog.getExistingDirectory()
        self.ImgSavePath.setText(filename)

    def SaveModel(self):
        if self.SaveBtn.isChecked():
            self.saveModel = 1
            self.ModelNameLabel.show()
            self.ModelPathLabel.show()
            self.BrowseMdlPath.show()
            self.ModelSavePath.show()
            self.ModelName.show()
        else:
            self.saveModel = 0
            self.ModelNameLabel.hide()
            self.ModelPathLabel.hide()
            self.ModelSavePath.hide()
            self.BrowseMdlPath.hide()
            self.ModelName.hide()

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
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if not self.tranData_Path.text().endswith(".shp"):
            msg.setText("Invalid Training Data File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText("The details are as follows: Either Training file is not selected or does not contain .shp extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if self.attributes.currentText() == "" :
            msg.setText("Please select the attribute. ")
            msg.setInformativeText("Attribute not selected")
            msg.setDetailedText("The details are as follows: Select attribute from Training File")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return


        if self.ReportName.text() == "":
            msg.setText("Please enter a suitable name for report. ")
            msg.setInformativeText("Report name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if self.ReportsavePath.text() == "":
            msg.setText("Please select path for report. ")
            msg.setInformativeText("Report path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if self.ImgName.text() == "":
            msg.setText("Please enter a suitable name for Image result. ")
            msg.setInformativeText("Image name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if self.ImgSavePath.text() == "":
            msg.setText("Please select path for Image result. ")
            msg.setInformativeText("Image path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if self.saveModel == 1:
            if self.ModelName.text() == "":
                msg.setText("Please enter name for saving model. ")
                msg.setInformativeText("Model name missing")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_RegressionWindow()
                self.ui.setupUi(self.window)
                return

            if self.ModelSavePath.text() == "":
                msg.setText("Please path for saving model. ")
                msg.setInformativeText("Model path missing")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_RegressionWindow()
                self.ui.setupUi(self.window)
                return

        else:
            self.Run()


    def Run(self):


        reportpath = self.ReportsavePath.text()
        reportpath+="/"+str(self.ReportName.text())+".pdf"

        prediction_map = self.ReportsavePath.text()
        prediction_map += "/" + str(self.ImgName.text())+"tif"

        modelname = self.ModelName.text()
        modelsavepath = self.ModelSavePath.text() + "/" + str(modelname)+".sav"



        LoadingImages = self.input_C.load_image_data()
        img_ds = LoadingImages[0]
        img = LoadingImages[1]

        doc = self.rt.build_doc(reportpath,"Regression")
        self.input_C.set_training_attr(self.attributes.currentText())

        TrainingData = self.input_C.load_train_data(img_ds)  ## roi

       # self.input_C.create_training_subplots(img[:, :, 0],TrainingData)

        Regressor = self.RgMdl.RF_regressor(TrainingData,img,self.attributes.currentText())
        RFR = Regressor[0]
        self.helper = Regressor[1]
        #preparesection1
        prediction = self.RgMdl.RF_prediction(RFR,img)

        self.RgMdl.RF_save_PredictionImage(prediction,prediction_map,img,img_ds)


        doc = self.rt.Reg_prepare_report(doc,img,prediction,TrainingData,self.helper,self.ImgPath.text(),self.tranData_Path.text(),reportpath,prediction_map,modelsavepath)

        if self.saveModel == 1:
            self.RgMdl.save_model(RFR,modelsavepath)

        doc.save()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Complete!!!")
        msg.setText("Processing Done. Repoort ready to preview ")
        msg.exec_()