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
    from RegressionController import RegressionController
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QFileDialog,QMessageBox
    from osgeo import ogr
except Exception as e:
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)



class Ui_RegressionWindow(object):
    input_C = InputController()
    RgMdl = RegressionController()
    rt = ReportModule()
    saveModel = 0

    def setupUi(self, RegressionWindow):
        RegressionWindow.setObjectName("RegressionWindow")
        RegressionWindow.resize(870, 845)
        self.centralwidget = QtWidgets.QWidget(RegressionWindow)
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
        self.label.setGeometry(QtCore.QRect(210, 30, 521, 61))
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
        self.testsize = QtWidgets.QLineEdit(self.centralwidget)
        self.testsize.setGeometry(QtCore.QRect(30, 400, 111, 41))
        self.testsize.setObjectName("testsize")
        self.trees_label = QtWidgets.QLabel(self.centralwidget)
        self.trees_label.setGeometry(QtCore.QRect(330, 340, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.trees_label.setFont(font)
        self.trees_label.setObjectName("trees_label")
        self.SplitSize_label = QtWidgets.QLabel(self.centralwidget)
        self.SplitSize_label.setGeometry(QtCore.QRect(20, 340, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.SplitSize_label.setFont(font)
        self.SplitSize_label.setObjectName("SplitSize_label")
        self.trnsplt_label = QtWidgets.QLabel(self.centralwidget)
        self.trnsplt_label.setGeometry(QtCore.QRect(150, 400, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.trnsplt_label.setFont(font)
        self.trnsplt_label.setObjectName("trnsplt_label")
        self.TreesNo = QtWidgets.QLineEdit(self.centralwidget)
        self.TreesNo.setGeometry(QtCore.QRect(330, 380, 113, 41))
        self.TreesNo.setObjectName("TreesNo")
        self.SplitSize_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.SplitSize_label_2.setGeometry(QtCore.QRect(20, 460, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.SplitSize_label_2.setFont(font)
        self.SplitSize_label_2.setObjectName("SplitSize_label_2")
        RegressionWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RegressionWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 870, 26))
        self.menubar.setObjectName("menubar")
        RegressionWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RegressionWindow)
        self.statusbar.setObjectName("statusbar")
        RegressionWindow.setStatusBar(self.statusbar)

        self.testsize.setText("25")
        self.TreesNo.setText("100")

        self.ImgPth_Btn.clicked.connect(self.openImagePathDialog)
        self.tranPath_Btn.clicked.connect(self.openTrainingFileDialog)

        self.BrowsePath.clicked.connect(self.BrowsePathDir)

        self.SaveBtn.toggled.connect(self.SaveModel)
        self.attr_btn.clicked.connect(self.Get_TrainAttribute)
        self.RunBtn.clicked.connect(self.validateInput)

        self.retranslateUi(RegressionWindow)
        QtCore.QMetaObject.connectSlotsByName(RegressionWindow)

    def retranslateUi(self, RegressionWindow):
        _translate = QtCore.QCoreApplication.translate
        RegressionWindow.setWindowTitle(_translate("RegressionWindow", "Regression"))
        self.ImgPth_Btn.setText(_translate("RegressionWindow", "Browse"))
        self.tranPath_Btn.setText(_translate("RegressionWindow", "Browse"))
        self.attr_btn.setText(_translate("RegressionWindow", "Get attributes"))
        self.RunBtn.setText(_translate("RegressionWindow", "Run"))
        self.label.setText(_translate("RegressionWindow", "Random Forrest Regression"))
        self.PathLabel.setText(_translate("RegressionWindow", "Results Directory"))
        self.BrowsePath.setText(_translate("RegressionWindow", "Browse"))
        self.SaveBtn.setText(_translate("RegressionWindow", "Save Model"))
        self.ReportNameLabel.setText(_translate("RegressionWindow", "Project Name"))
        self.ImgPath_label.setText(_translate("RegressionWindow", "Image Path"))
        self.TrainData_label.setText(_translate("RegressionWindow", "Training Data Path"))
        self.trees_label.setText(_translate("RegressionWindow", "Number of Trees"))
        self.SplitSize_label.setText(_translate("RegressionWindow", "Split size of Training & Testing sets"))
        self.trnsplt_label.setText(_translate("RegressionWindow", "% Test Size"))
        self.SplitSize_label_2.setText(_translate("RegressionWindow", "Training Size is 100%  -  Test Size"))


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


        if self.ProjectName.text() == "":
            msg.setText("Please enter a suitable name for your project. ")
            msg.setInformativeText("project name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        if self.savePath.text() == "":
            msg.setText("Please select path for project. ")
            msg.setInformativeText("project path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RegressionWindow()
            self.ui.setupUi(self.window)
            return

        # if self.TreesNo.text() == "":
        #     msg.setText("Please enter a suitable numbe. ")
        #     msg.setInformativeText("Image name missing")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_RegressionWindow()
        #     self.ui.setupUi(self.window)
        #     return
        #
        # if self.ImgSavePath.text() == "":
        #     msg.setText("Please select path for Image result. ")
        #     msg.setInformativeText("Image path missing")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_RegressionWindow()
        #     self.ui.setupUi(self.window)
        #     return
        #
        # if self.saveModel == 1 and (self.ModelName.text() == "" or self.ModelSavePath.text() == ""):
        #     msg.setText("Please enter name and Path for saving model. ")
        #     msg.setInformativeText("Either Model Name of Path not entered")
        #     msg.exec_()
        #     self.window = QtWidgets.QMainWindow()
        #     self.ui = Ui_RegressionWindow()
        #     self.ui.setupUi(self.window)
        #     return

        else:
            self.Run()


    def Run(self):

        try:
            print("The process has started.... This will take few minutes")
            if self.TreesNo.text() == "":
                self.TreesNo.setText("100")

            if self.testsize.text() == "":
                self.testsize.setText("1000")

            test_size = float(self.testsize.text())/100
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
                self.ui = Ui_RegressionWindow()
                self.ui.setupUi(self.window)
                return

            else:
                os.mkdir(dir_path)

            reportpath = dir_path
            reportpath+="/"+str(self.ProjectName.text())+"_REPORT.pdf"

            prediction_map = dir_path
            prediction_map += "/" + str(self.ProjectName.text())+"_IMG.tif"
            modelsavepath = dir_path+"/"+self.ProjectName.text() + "_MODEL.sav"

            LoadingImages = self.input_C.load_image_data()
            img_ds = LoadingImages[0]
            img = LoadingImages[1]

            doc = self.rt.build_doc(reportpath,"Random Forrest Regression Report")
            self.input_C.set_training_attr(self.attributes.currentText())

            TrainingData = self.input_C.load_train_data(img_ds,"Regression")  ## roi

           # self.input_C.create_training_subplots(img[:, :, 0],TrainingData)

            Regressor = self.RgMdl.RF_regressor(TrainingData,img,self.attributes.currentText(),test_size,treeNo)
            RFR = Regressor[0]
            self.helper = Regressor[1]
            #preparesection1
            prediction = self.RgMdl.RF_prediction(RFR,img)

            self.RgMdl.RF_save_PredictionImage(prediction,prediction_map,img,img_ds)

            rp_param = RFHelper(prediction,TrainingData,self.helper,self.ImgPath.text(),self.tranData_Path.text(),reportpath,prediction_map,modelsavepath)


            doc = self.rt.make_rf_reg_report(doc,img,rp_param,dir_path)

            if self.saveModel == 1:
                self.RgMdl.save_model(RFR,modelsavepath)

            doc.save()

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Complete!!!")
            msg.setText("Processing Done. Report ready to preview ")
            msg.exec_()

        except ValueError as e:
            print(e)