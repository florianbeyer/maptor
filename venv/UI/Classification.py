try:
    import sys,os,logging
    from InputController import InputController
    from ClassificationController import ClassificationController
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QFileDialog,QMessageBox
    from osgeo import ogr
    from ReportModule import ReportModule
except Exception as e:
    print("Cannot import"+str(e))
    input("Press Enter to exit!")
    sys.exit(0)


class Ui_ClassificationWindow(object):
    input_C = InputController()
    rf_C = ClassificationController()
    rt = ReportModule()
    saveModel = 0
    counter = 0
    validationFlag = 1

    def setupUi(self, ClassificationWindow):
        ClassificationWindow.setObjectName("ClassificationWindow")
        ClassificationWindow.resize(860, 798)
        self.centralwidget = QtWidgets.QWidget(ClassificationWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imagePath = QtWidgets.QLineEdit(self.centralwidget)
        self.imagePath.setGeometry(QtCore.QRect(20, 140, 311, 31))
        self.imagePath.setObjectName("imagePath")
        self.trainPath = QtWidgets.QLineEdit(self.centralwidget)
        self.trainPath.setGeometry(QtCore.QRect(20, 230, 311, 31))
        self.trainPath.setObjectName("trainPath")
        self.validPath = QtWidgets.QLineEdit(self.centralwidget)
        self.validPath.setGeometry(QtCore.QRect(20, 320, 311, 31))
        self.validPath.setObjectName("validPath")
        self.imgBrowseBtn = QtWidgets.QPushButton(self.centralwidget)
        self.imgBrowseBtn.setGeometry(QtCore.QRect(370, 140, 93, 31))
        self.imgBrowseBtn.setObjectName("imgBrowseBtn")
        self.trainBrowseBtn = QtWidgets.QPushButton(self.centralwidget)
        self.trainBrowseBtn.setGeometry(QtCore.QRect(370, 230, 93, 31))
        self.trainBrowseBtn.setObjectName("trainBrowseBtn")
        self.validBrowseBtn = QtWidgets.QPushButton(self.centralwidget)
        self.validBrowseBtn.setGeometry(QtCore.QRect(370, 320, 93, 31))
        self.validBrowseBtn.setObjectName("validBrowseBtn")
        self.ImagePathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImagePathLabel.setGeometry(QtCore.QRect(30, 110, 131, 20))
        self.ImagePathLabel.setObjectName("ImagePathLabel")
        self.TrainingPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.TrainingPathLabel.setGeometry(QtCore.QRect(30, 200, 131, 20))
        self.TrainingPathLabel.setObjectName("TrainingPathLabel")
        self.ValdationPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ValdationPathLabel.setGeometry(QtCore.QRect(30, 290, 131, 20))
        self.ValdationPathLabel.setObjectName("ValdationPathLabel")
        self.TreesLabel = QtWidgets.QLabel(self.centralwidget)
        self.TreesLabel.setGeometry(QtCore.QRect(30, 400, 131, 20))
        self.TreesLabel.setObjectName("TreesLabel")
        self.ClassificationLabel = QtWidgets.QLabel(self.centralwidget)
        self.ClassificationLabel.setGeometry(QtCore.QRect(160, 10, 611, 61))
        font = QtGui.QFont()
        font.setPointSize(25)
        font.setBold(False)
        font.setWeight(50)
        self.ClassificationLabel.setFont(font)
        self.ClassificationLabel.setObjectName("ClassificationLabel")
        self.trainingAttributes = QtWidgets.QComboBox(self.centralwidget)
        self.trainingAttributes.setGeometry(QtCore.QRect(680, 240, 151, 31))
        self.trainingAttributes.setObjectName("trainingAttributes")
        self.validationAttributes = QtWidgets.QComboBox(self.centralwidget)
        self.validationAttributes.setGeometry(QtCore.QRect(680, 320, 151, 31))
        self.validationAttributes.setObjectName("validationAttributes")
        self.trainAttrBtn = QtWidgets.QPushButton(self.centralwidget)
        self.trainAttrBtn.setGeometry(QtCore.QRect(570, 240, 93, 28))
        self.trainAttrBtn.setObjectName("trainAttrBtn")
        self.valAttrBtn = QtWidgets.QPushButton(self.centralwidget)
        self.valAttrBtn.setGeometry(QtCore.QRect(570, 320, 93, 28))
        self.valAttrBtn.setObjectName("valAttrBtn")
        self.savePath = QtWidgets.QLineEdit(self.centralwidget)
        self.savePath.setGeometry(QtCore.QRect(20, 560, 321, 31))
        self.savePath.setObjectName("savePath")
        self.ReportPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportPathLabel.setGeometry(QtCore.QRect(20, 530, 131, 20))
        self.ReportPathLabel.setObjectName("ReportPathLabel")
        self.BrowsePath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowsePath.setGeometry(QtCore.QRect(360, 560, 93, 31))
        self.BrowsePath.setObjectName("BrowsePath")
        self.runBtn = QtWidgets.QPushButton(self.centralwidget)
        self.runBtn.setGeometry(QtCore.QRect(710, 440, 121, 41))
        self.runBtn.setObjectName("runBtn")
        self.treesNo = QtWidgets.QLineEdit(self.centralwidget)
        self.treesNo.setGeometry(QtCore.QRect(20, 440, 131, 41))
        self.treesNo.setObjectName("treesNo")
        self.SaveBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveBtn.setGeometry(QtCore.QRect(600, 530, 93, 20))
        self.SaveBtn.setObjectName("SaveBtn")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 500, 1061, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.ProjectName = QtWidgets.QLineEdit(self.centralwidget)
        self.ProjectName.setGeometry(QtCore.QRect(20, 630, 221, 31))
        self.ProjectName.setObjectName("ReportName")
        self.ProjectNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ProjectNameLabel.setGeometry(QtCore.QRect(20, 610, 131, 20))
        self.ProjectNameLabel.setObjectName("ReportNameLabel")
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

        self.SaveBtn.toggled.connect(self.SaveModel)
        self.BrowsePath.clicked.connect(self.BrowsePathDir)
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
        self.ClassificationLabel.setText(_translate("ClassificationWindow", "Random Forrest Classification "))
        self.trainAttrBtn.setText(_translate("ClassificationWindow", "Get Attributes"))
        self.valAttrBtn.setText(_translate("ClassificationWindow", "Get Attributes"))
        self.ReportPathLabel.setText(_translate("ClassificationWindow", "Results Directory"))
        self.BrowsePath.setText(_translate("ClassificationWindow", "Browse"))
        self.runBtn.setText(_translate("ClassificationWindow", "Run"))
        self.SaveBtn.setText(_translate("ClassificationWindow", "Save Model"))
        self.ProjectNameLabel.setText(_translate("ClassificationWindow", "Project Name"))

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
            if(attlist):
                self.trainingAttributes.addItems(attlist)
            if not(attlist):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("NO ATTRIBUTES FOUND")
                msg.setText("NO ATTRIBUTES FOUND IN .SHP FILE. ")
                msg.exec_()

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
            if (attlist):
                self.validationAttributes.addItems(attlist)
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

    def loadImage(self):
        self.input_C.load_image_data()

    def BrowsePathDir(self):
        filename = QFileDialog.getExistingDirectory()
        self.savePath.setText(filename)

    # def BrowseReportPath(self):
    #     filename = QFileDialog.getExistingDirectory()
    #     self.ReportsavePath.setText(filename)

    # def BrowseImagePath(self):
    #     filename = QFileDialog.getExistingDirectory()
    #     self.ImgSavePath.setText(filename)

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

    # def LoadValidationDialog(self):
    #     self.window = QtWidgets.QMainWindow()
    #     self.ui = Ui_ValidationDialog()
    #     self.ui.setupUi(self.window)
    #     # MainWindow.hide()
    #     self.window.show()


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

        if self.ProjectName.text() == "":
            msg.setText("Please enter a suitable name for your project. ")
            msg.setInformativeText("Project name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        if self.savePath.text() == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Validation Prompt")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            msg.setText("Please select path for Results. ")
            msg.setInformativeText("Results path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_ClassificationWindow()
            self.ui.setupUi(self.window)
            return

        # if self.ImgName.text() == "":
        #     msg.setText("Please enter a suitable name for Image result. ")
        #     msg.setInformativeText("Image name missing")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_ClassificationWindow()
        #     self.ui.setupUi(self.window)
        #     return
        #
        # if self.ImgSavePath.text() == "":
        #     msg.setText("Please select path for Image result. ")
        #     msg.setInformativeText("Image path missing")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_ClassificationWindow()
        #     self.ui.setupUi(self.window)
        #     return

        # if self.saveModel == 1 and (self.ModelName.text() == "" or self.ModelSavePath.text() == ""):
        #     msg.setText("Please enter name and Path for saving model. ")
        #     msg.setInformativeText("Either Model Name of Path not entered")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_ClassificationWindow()
        #     self.ui.setupUi(self.window)
        #     return

        else:
            self.Run()

    def Run(self):
        try:
            # import sys, os
            # try:
            #     ROOT_DIR = os.path.abspath(os.curdir)
            #     PROJ_DIR = ROOT_DIR + "\PROJ"
            #     os.environ['PROJ_LIB'] = PROJ_DIR
            #
            #     if 'PROJ_LIB' in os.environ:
            #         print('env variable : PROJ_LIB  set...')
            #     else:
            #         print('Couldnt set env variable : PROJ_LIB.Please set Manually ')
            # except Exception as ex:
            #     print("Could not set env_var")

            print("The process has started.... This will take few minutes")
            if self.treesNo.text() == "":
                trees = 100
            else:
                trees = int(self.treesNo.text())

            self.rf_C.set_RF_trees = trees

            print (self.savePath.text())
            dir_path = self.savePath.text()+"\\"+self.ProjectName.text()

            if os.path.isdir(dir_path):

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Validation Prompt")
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                msg.setText("A folder nammed "+ self.ProjectName.text()+" already exists. ")
                msg.setInformativeText("Please move the folder and try again")
                msg.exec_()
                self.window = QtWidgets.QMainWindow()
                self.ui = Ui_ClassificationWindow()
                self.ui.setupUi(self.window)
                return

            else:
                os.mkdir(dir_path)
        #
            doc_path = dir_path + "/" + str(self.ProjectName.text()) +"_REPORT.pdf"
            print(doc_path)
        #
            doc = self.rt.build_doc(doc_path,"Random Forrest Classification Report")
            LoadingImages = self.input_C.load_image_data()
            img_ds = LoadingImages[0]
            img = LoadingImages[1]
        #
        #
            self.input_C.set_training_attr(self.trainingAttributes.currentText())
            self.input_C.set_validation_attr(self.validationAttributes.currentText())

            TrainingData = self.input_C.load_train_data(img_ds,"Classification")  ## roi
            ValidationData = self.input_C.load_validation_data(img_ds)  ##roi_V

            classifier_output = self.rf_C.rf_classifier(img, img_ds, TrainingData, trees)

            model = classifier_output[0]
            importance = classifier_output[1]
            table_M = classifier_output[2]
            ob_score = classifier_output[3]


            if self.SaveBtn.isChecked() == True:
                model_path = dir_path
                model_path += "/" + str(self.ProjectName.text()) + "_MODEL.sav"
                self.rf_C.save_model(model, model_path)
            else:
                model_path = "Model not saved"

        #
            class_prediction = self.rf_C.rf_prediction(img, model)
            imgSavePath = dir_path
            imgSavePath += "/"+ (self.ProjectName.text())+"_IMG.tif"
            self.rf_C.save_result_image(img_ds, img, class_prediction, imgSavePath)

            doc = self.rt.Clf_prepare_report(doc, img, TrainingData, importance, table_M, trees, ob_score, class_prediction,
                                           ValidationData,self.trainingAttributes.currentText(),dir_path)

            doc = self.rf_C.validation_processing(ValidationData, class_prediction, TrainingData, doc,model_path,imgSavePath,dir_path)
            doc.save()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Complete!!!")
            msg.setText("Processing Done. Report ready to preview ")
            msg.exec_()

        except ValueError:
            logging.error("Exception occurred", exc_info=True)
            print("Classification Failed...")
