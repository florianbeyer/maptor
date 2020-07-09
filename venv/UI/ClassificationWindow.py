import sys
import os
sys.path.append(r"F:\Work\Maptor\maptor\venv\Model")
sys.path.append(r"F:\Work\Maptor\maptor\venv\Controller")
from InputController import InputController
from RFController import RandomForrestController
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QInputDialog, QLineEdit, QFileDialog,QDialog,QMessageBox
from osgeo import ogr
from osgeo import gdal, ogr, gdal_array
from ReportModule import ReportModule
import joblib
import time


class Ui_ClassificationWindow(object):

    input_C = InputController()
    rf_C = RandomForrestController()
    rt = ReportModule()
    saveModel = 0
    counter = 0
    validationFlag = 1

    def setupUi(self, ClassificationWindow):
        ClassificationWindow.setObjectName("ClassificationWindow")
        ClassificationWindow.resize(860, 895)
        self.centralwidget = QtWidgets.QWidget(ClassificationWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imagePath = QtWidgets.QLineEdit(self.centralwidget)
        self.imagePath.setGeometry(QtCore.QRect(20, 120, 311, 31))
        self.imagePath.setObjectName("imagePath")
        self.trainPath = QtWidgets.QLineEdit(self.centralwidget)
        self.trainPath.setGeometry(QtCore.QRect(20, 210, 311, 31))
        self.trainPath.setObjectName("trainPath")
        self.validPath = QtWidgets.QLineEdit(self.centralwidget)
        self.validPath.setGeometry(QtCore.QRect(20, 300, 311, 31))
        self.validPath.setObjectName("validPath")
        self.imgBrowseBtn = QtWidgets.QPushButton(self.centralwidget)
        self.imgBrowseBtn.setGeometry(QtCore.QRect(370, 120, 93, 31))
        self.imgBrowseBtn.setObjectName("imgBrowseBtn")
        self.trainBrowseBtn = QtWidgets.QPushButton(self.centralwidget)
        self.trainBrowseBtn.setGeometry(QtCore.QRect(370, 210, 93, 31))
        self.trainBrowseBtn.setObjectName("trainBrowseBtn")
        self.validBrowseBtn = QtWidgets.QPushButton(self.centralwidget)
        self.validBrowseBtn.setGeometry(QtCore.QRect(370, 300, 93, 31))
        self.validBrowseBtn.setObjectName("validBrowseBtn")
        self.ImagePathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImagePathLabel.setGeometry(QtCore.QRect(30, 90, 131, 20))
        self.ImagePathLabel.setObjectName("ImagePathLabel")
        self.TrainingPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.TrainingPathLabel.setGeometry(QtCore.QRect(30, 180, 131, 20))
        self.TrainingPathLabel.setObjectName("TrainingPathLabel")
        self.ValdationPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ValdationPathLabel.setGeometry(QtCore.QRect(30, 270, 131, 20))
        self.ValdationPathLabel.setObjectName("ValdationPathLabel")
        self.TreesLabel = QtWidgets.QLabel(self.centralwidget)
        self.TreesLabel.setGeometry(QtCore.QRect(30, 380, 131, 20))
        self.TreesLabel.setObjectName("TreesLabel")
        self.ClassificationLabel = QtWidgets.QLabel(self.centralwidget)
        self.ClassificationLabel.setGeometry(QtCore.QRect(350, 10, 291, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(False)
        font.setWeight(50)
        self.ClassificationLabel.setFont(font)
        self.ClassificationLabel.setObjectName("ClassificationLabel")
        self.trainingAttributes = QtWidgets.QComboBox(self.centralwidget)
        self.trainingAttributes.setGeometry(QtCore.QRect(680, 220, 151, 31))
        self.trainingAttributes.setObjectName("trainingAttributes")
        self.validationAttributes = QtWidgets.QComboBox(self.centralwidget)
        self.validationAttributes.setGeometry(QtCore.QRect(680, 300, 151, 31))
        self.validationAttributes.setObjectName("validationAttributes")
        self.trainAttrBtn = QtWidgets.QPushButton(self.centralwidget)
        self.trainAttrBtn.setGeometry(QtCore.QRect(570, 220, 93, 28))
        self.trainAttrBtn.setObjectName("trainAttrBtn")
        self.valAttrBtn = QtWidgets.QPushButton(self.centralwidget)
        self.valAttrBtn.setGeometry(QtCore.QRect(570, 300, 93, 28))
        self.valAttrBtn.setObjectName("valAttrBtn")
        self.ReportsavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.ReportsavePath.setGeometry(QtCore.QRect(20, 620, 221, 31))
        self.ReportsavePath.setObjectName("ReportsavePath")
        self.ReportPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportPathLabel.setGeometry(QtCore.QRect(20, 590, 131, 20))
        self.ReportPathLabel.setObjectName("ReportPathLabel")
        self.BrowseRptPath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseRptPath.setGeometry(QtCore.QRect(260, 620, 93, 31))
        self.BrowseRptPath.setObjectName("BrowseRptPath")
        self.runBtn = QtWidgets.QPushButton(self.centralwidget)
        self.runBtn.setGeometry(QtCore.QRect(710, 420, 121, 41))
        self.runBtn.setObjectName("runBtn")
        self.treesNo = QtWidgets.QLineEdit(self.centralwidget)
        self.treesNo.setGeometry(QtCore.QRect(20, 420, 131, 41))
        self.treesNo.setObjectName("treesNo")
        self.SaveBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveBtn.setGeometry(QtCore.QRect(600, 500, 111, 21))
        self.SaveBtn.setObjectName("SaveBtn")
        self.ModelSavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.ModelSavePath.setGeometry(QtCore.QRect(600, 660, 221, 31))
        self.ModelSavePath.setObjectName("ModelSavePath")
        self.BrowseMdlPath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseMdlPath.setGeometry(QtCore.QRect(730, 710, 93, 31))
        self.BrowseMdlPath.setObjectName("BrowseMdlPath")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 480, 1061, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.ModelPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ModelPathLabel.setGeometry(QtCore.QRect(600, 620, 131, 20))
        self.ModelPathLabel.setObjectName("ModelPathLabel")
        self.ModelName = QtWidgets.QLineEdit(self.centralwidget)
        self.ModelName.setGeometry(QtCore.QRect(600, 570, 221, 31))
        self.ModelName.setObjectName("ModelName")
        self.ModelNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ModelNameLabel.setGeometry(QtCore.QRect(600, 540, 211, 20))
        self.ModelNameLabel.setObjectName("ModelNameLabel")
        self.ReportName = QtWidgets.QLineEdit(self.centralwidget)
        self.ReportName.setGeometry(QtCore.QRect(20, 530, 221, 31))
        self.ReportName.setObjectName("ReportName")
        self.ReportNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportNameLabel.setGeometry(QtCore.QRect(20, 500, 131, 20))
        self.ReportNameLabel.setObjectName("ReportNameLabel")
        self.ImgSavePath = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgSavePath.setGeometry(QtCore.QRect(20, 790, 221, 31))
        self.ImgSavePath.setObjectName("ImgSavePath")
        self.ImgName = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgName.setGeometry(QtCore.QRect(20, 700, 221, 31))
        self.ImgName.setObjectName("ImgName")
        self.ImgNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgNameLabel.setGeometry(QtCore.QRect(20, 670, 161, 20))
        self.ImgNameLabel.setObjectName("ImgNameLabel")
        self.ImgPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgPathLabel.setGeometry(QtCore.QRect(20, 760, 131, 20))
        self.ImgPathLabel.setObjectName("ImgPathLabel")
        self.BrowseImgSavetBtn = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImgSavetBtn.setGeometry(QtCore.QRect(260, 790, 93, 31))
        self.BrowseImgSavetBtn.setObjectName("BrowseImgSavetBtn")
        ClassificationWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ClassificationWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 860, 26))
        self.menubar.setObjectName("menubar")
        ClassificationWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ClassificationWindow)
        self.statusbar.setObjectName("statusbar")
        ClassificationWindow.setStatusBar(self.statusbar)

        self.treesNo.setText("500")

        self.validBrowseBtn.clicked.connect(self.openValidationDialog)
        self.trainBrowseBtn.clicked.connect(self.openTrainingDialog)
        self.imgBrowseBtn.clicked.connect(self.openImageDialog)

        self.trainAttrBtn.clicked.connect(self.Get_TrainAttribute)
        self.valAttrBtn.clicked.connect(self.Get_ValidationAttribute)

        self.ModelNameLabel.hide()
        self.ModelPathLabel.hide()
        self.BrowseMdlPath.hide()
        self.ModelName.hide()
        self.ModelSavePath.hide()

        self.SaveBtn.toggled.connect(self.SaveModel)

        self.BrowseMdlPath.clicked.connect(self.BrowseModelPath)

        self.BrowseRptPath.clicked.connect(self.BrowseReportPath)
        self.BrowseImgSavetBtn.clicked.connect(self.BrowseImagePath)

        self.runBtn.clicked.connect(self.validateInput)

        self.retranslateUi(ClassificationWindow)
        QtCore.QMetaObject.connectSlotsByName(ClassificationWindow)

    def retranslateUi(self, ClassificationWindow):
        _translate = QtCore.QCoreApplication.translate
        ClassificationWindow.setWindowTitle(_translate("ClassificationWindow", "MainWindow"))
        self.imgBrowseBtn.setText(_translate("ClassificationWindow", "Browse"))
        self.trainBrowseBtn.setText(_translate("ClassificationWindow", "Browse"))
        self.validBrowseBtn.setText(_translate("ClassificationWindow", "Browse"))
        self.ImagePathLabel.setText(_translate("ClassificationWindow", "Image Path"))
        self.TrainingPathLabel.setText(_translate("ClassificationWindow", "Training Path"))
        self.ValdationPathLabel.setText(_translate("ClassificationWindow", "Validation Path"))
        self.TreesLabel.setText(_translate("ClassificationWindow", "Number of Trees"))
        self.ClassificationLabel.setText(_translate("ClassificationWindow", " Classification "))
        self.trainAttrBtn.setText(_translate("ClassificationWindow", "Get Attributes"))
        self.valAttrBtn.setText(_translate("ClassificationWindow", "Get Attributes"))
        self.ReportPathLabel.setText(_translate("ClassificationWindow", "Save Report to Path"))
        self.BrowseRptPath.setText(_translate("ClassificationWindow", "Browse"))
        self.runBtn.setText(_translate("ClassificationWindow", "Run"))
        self.SaveBtn.setText(_translate("ClassificationWindow", "Save Model"))
        self.BrowseMdlPath.setText(_translate("ClassificationWindow", "Browse"))
        self.ModelPathLabel.setText(_translate("ClassificationWindow", "Save Model to Path"))
        self.ModelNameLabel.setText(_translate("ClassificationWindow", "Model Name (with .sav) extension"))
        self.ReportNameLabel.setText(_translate("ClassificationWindow", "Report Name"))
        self.ImgNameLabel.setText(_translate("ClassificationWindow", "Save Result Image Name"))
        self.ImgPathLabel.setText(_translate("ClassificationWindow", "Result Image Path"))
        self.BrowseImgSavetBtn.setText(_translate("ClassificationWindow", "Browse"))


    def openValidationDialog(self):
        filename = QFileDialog.getOpenFileName()
        self.validPath.setText(filename[0])
        self.input_C.set_validation_path(filename[0])

    def openTrainingDialog(self):
        filename = QFileDialog.getOpenFileName()
        self.trainPath.setText(filename[0])
        self.input_C.set_training_path(filename[0])

    def openImageDialog(self):
        filename = QFileDialog.getOpenFileName()
        self.imagePath.setText(filename[0])
        self.input_C.set_img_path(filename[0])

    def Get_TrainAttribute(self):
        if not self.trainPath.text().endswith(".shp"):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Wrong Selection")
            msg.setText("Please select correct file with .shp extension to get attributes. ")
            msg.exec_()
            return

        else:
            attlist = self.FindAttributes(self.trainPath.text())
            self.trainingAttributes.addItems(attlist)

    def Get_ValidationAttribute(self):
        if not self.validPath.text().endswith(".shp"):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Wrong Selection")
            msg.setText("Please select correct file with .shp extension to get attributes. ")
            msg.exec_()
            return
        else:
            attlist = self.FindAttributes(self.validPath.text())
            self.validationAttributes.addItems(attlist)

    def FindAttributes(self, filepath):
        try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            shape_dataset = driver.Open(filepath)
            shape_layer = shape_dataset.GetLayer()
            field_names = [field.name for field in shape_layer.schema]
            return field_names
        except ValueError as e:
            print(e)

    def loadImage(self):
        self.input_C.load_image_data()

    # //def OpenPopUp(self):
    #     self.window = QtWidgets.QMainWindow()
    #     self.ui = Ui_PopUp()
    #     self.ui.setupUi(self.window)
    #     # MainWindow.hide()
    #     self.window.show()

    def BrowseModelPath(self):
        filename = QFileDialog.getExistingDirectory()
        self.ModelSavePath.setText(filename)

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

    def LoadValidationDialog(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_ValidationDialog()
        self.ui.setupUi(self.window)
        # MainWindow.hide()
        self.window.show()


    def validateInput(self):

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Validation Prompt")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if not self.imagePath.text().endswith(".tif"):
            msg.setText("Invalid Image File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText("The details are as follows: Either Image file is not selected or does not contain .tif extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if not self.trainPath.text().endswith(".shp"):
            msg.setText("Invalid Training Data File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText("The details are as follows: Either Training file is not selected or does not contain .shp extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if not self.validPath.text().endswith(".shp"):
            msg.setText("Invalid Validation Data File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText("The details are as follows: Either Validation file is not selected or does not contain .shp extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if self.validationAttributes.currentText() == "" or self.trainingAttributes.currentText() == "":
            msg.setText("Please select the attributes. ")
            msg.setInformativeText("Attribute not selected")
            msg.setDetailedText("The details are as follows: Select same attribute from Training and Validation File")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if not self.validationAttributes.currentText() == "" or self.trainingAttributes.currentText() == "":
            if self.validationAttributes.currentText() != self.trainingAttributes.currentText():
                msg.setText("Invalid Data. ")
                msg.setInformativeText("Please see details for Error Message")
                msg.setDetailedText(
                    "The details are as follows: Select same attribute from Training and Validation File")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_ClassificationWindow()
                self.ui.setupUi(self.window)
                return

        if self.ReportName.text() == "":
            msg.setText("Please enter a suitable name for report. ")
            msg.setInformativeText("Report name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if self.ReportsavePath.text() == "":
            msg.setText("Please select path for report. ")
            msg.setInformativeText("Report path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if self.ImgName.text() == "":
            msg.setText("Please enter a suitable name for Image result. ")
            msg.setInformativeText("Image name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if self.ImgSavePath.text() == "":
            msg.setText("Please select path for Image result. ")
            msg.setInformativeText("Image path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if self.saveModel == 1:
            if self.ModelName.text() == "":
                msg.setText("Please enter name for saving model. ")
                msg.setInformativeText("Model name missing")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_ClassificationWindow()
                self.ui.setupUi(self.window)
                return

            if self.ModelSavePath.text() == "":
                msg.setText("Please path for saving model. ")
                msg.setInformativeText("Model path missing")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_ClassificationWindow()
                self.ui.setupUi(self.window)
                return

        else:
            self.Run()

    def Run(self):
        try:
            if self.treesNo.text() == "":
                trees = 100
            else:
                trees = int(self.treesNo.text())

            self.rf_C.set_RF_trees = trees

            doc_path = str(self.ReportsavePath.text()) + "/" + str(self.ReportName.text()) +".pdf"
            print(doc_path)

            doc = self.rt.build_doc(doc_path,"Classification")
            LoadingImages = self.input_C.load_image_data()
            img_ds = LoadingImages[0]
            img = LoadingImages[1]


            self.input_C.set_training_attr(self.trainingAttributes.currentText())
            self.input_C.set_validation_attr(self.validationAttributes.currentText())

            TrainingData = self.input_C.load_train_data(img_ds)  ## roi
            ValidationData = self.input_C.load_validation_data(img_ds)  ##roi_V

            classifier_output = self.rf_C.rf_classifier(img, img_ds, TrainingData, trees)

            model = classifier_output[0]
            importance = classifier_output[1]
            table_M = classifier_output[2]
            ob_score = classifier_output[3]

            model_path = str(self.ModelSavePath.text())
            model_path += "/" + str(self.ModelName.text()) +".sav"
            print(model_path)

            if self.saveModel == 1:
                self.rf_C.save_model(model, model_path)

            class_prediction = self.rf_C.rf_prediction(img, model)


            imgSavePath = str(self.ImgSavePath.text())

            imgSavePath += "/"+ (self.ImgName.text())+".tif"
            self.rf_C.save_result_image(img_ds, img, class_prediction, imgSavePath)

            doc = self.rt.Clf_prepare_report(doc, img, TrainingData, importance, table_M, trees, ob_score, class_prediction,
                                           ValidationData,self.trainingAttributes.currentText())
            doc = self.rf_C.validation_processing(ValidationData, class_prediction, TrainingData, doc,model_path,imgSavePath)
            doc.save()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Complete!!!")
            msg.setText("Processing Done. Repoort ready to preview ")
            msg.exec_()

        except ValueError:
            print("Classification Failed...")







