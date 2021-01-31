try:
    import sys,os,logging
    sys.path.append(r"..\Model")
    sys.path.append(r"..\HelpingModel")
    sys.path.append(r"..\Controller")
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    import matplotlib.pyplot as plt # plot figures
    import numpy as np
    from InputController import InputController
    from PLSR_SSS_Controller import PLSR_SSS_Controller
    from ReportModule import ReportModule
    from PLSR_SSS_Helper import PLSR_SSS_Helper
    import pandas as pd
    import traceback
except Exception as e:
    logging.error("Exception occurred", exc_info=True)
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)


class Ui_PLSR_SSS(object):
    input_C = InputController()
    PLSR_C = PLSR_SSS_Controller()
    rt = ReportModule()


    def setupUi(self, PLSR_Window):
        PLSR_Window.setObjectName("PLSR_Window")
        PLSR_Window.resize(886, 725)
        self.centralwidget = QtWidgets.QWidget(PLSR_Window)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowsePath = QtWidgets.QPushButton(self.centralwidget)
        self.BrowsePath.setGeometry(QtCore.QRect(410, 480, 93, 31))
        self.BrowsePath.setObjectName("BrowsePath")
        self.TrainData_label = QtWidgets.QLabel(self.centralwidget)
        self.TrainData_label.setGeometry(QtCore.QRect(30, 250, 131, 20))
        self.TrainData_label.setObjectName("TrainData_label")
        self.ImgPth_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.ImgPth_Btn.setGeometry(QtCore.QRect(390, 200, 93, 31))
        self.ImgPth_Btn.setObjectName("ImgPth_Btn")
        self.savePath = QtWidgets.QLineEdit(self.centralwidget)
        self.savePath.setGeometry(QtCore.QRect(50, 480, 321, 31))
        self.savePath.setObjectName("savePath")
        self.PathLabel = QtWidgets.QLabel(self.centralwidget)
        self.PathLabel.setGeometry(QtCore.QRect(50, 450, 131, 20))
        self.PathLabel.setObjectName("PathLabel")
        self.attr_btn = QtWidgets.QPushButton(self.centralwidget)
        self.attr_btn.setGeometry(QtCore.QRect(700, 230, 93, 31))
        self.attr_btn.setObjectName("attr_btn")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 390, 861, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(260, 30, 521, 61))
        font = QtGui.QFont()
        font.setPointSize(23)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.ImgPath_label = QtWidgets.QLabel(self.centralwidget)
        self.ImgPath_label.setGeometry(QtCore.QRect(30, 170, 131, 20))
        self.ImgPath_label.setObjectName("ImgPath_label")
        self.SaveBtn = QtWidgets.QRadioButton(self.centralwidget)
        self.SaveBtn.setGeometry(QtCore.QRect(750, 460, 111, 21))
        self.SaveBtn.setObjectName("SaveBtn")
        self.tranPath_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.tranPath_Btn.setGeometry(QtCore.QRect(390, 280, 93, 31))
        self.tranPath_Btn.setObjectName("tranPath_Btn")
        self.ReportNameLabel = QtWidgets.QLabel(self.centralwidget)
        self.ReportNameLabel.setGeometry(QtCore.QRect(50, 540, 131, 20))
        self.ReportNameLabel.setObjectName("ReportNameLabel")
        self.ImgPath = QtWidgets.QLineEdit(self.centralwidget)
        self.ImgPath.setGeometry(QtCore.QRect(30, 200, 311, 31))
        self.ImgPath.setObjectName("ImgPath")
        self.ProjectName = QtWidgets.QLineEdit(self.centralwidget)
        self.ProjectName.setGeometry(QtCore.QRect(50, 570, 221, 31))
        self.ProjectName.setObjectName("ProjectName")
        self.RunBtn = QtWidgets.QPushButton(self.centralwidget)
        self.RunBtn.setGeometry(QtCore.QRect(670, 310, 121, 41))
        self.RunBtn.setObjectName("RunBtn")
        self.tranData_Path = QtWidgets.QLineEdit(self.centralwidget)
        self.tranData_Path.setGeometry(QtCore.QRect(30, 280, 311, 31))
        self.tranData_Path.setObjectName("tranData_Path")
        self.attributes = QtWidgets.QComboBox(self.centralwidget)
        self.attributes.setGeometry(QtCore.QRect(640, 180, 151, 31))
        self.attributes.setObjectName("attributes")
        PLSR_Window.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PLSR_Window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 886, 26))
        self.menubar.setObjectName("menubar")
        PLSR_Window.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(PLSR_Window)
        self.statusbar.setObjectName("statusbar")
        PLSR_Window.setStatusBar(self.statusbar)

        self.ImgPth_Btn.clicked.connect(self.openImagePathDialog)
        self.attr_btn.clicked.connect(self.Get_TrainAttribute)
        self.tranPath_Btn.clicked.connect(self.openTrainingFileDialog)
        self.BrowsePath.clicked.connect(self.BrowsePathDir)
        self.RunBtn.clicked.connect(self.validateInput)

        self.retranslateUi(PLSR_Window)
        QtCore.QMetaObject.connectSlotsByName(PLSR_Window)

    def retranslateUi(self, PLSR_Window):
        _translate = QtCore.QCoreApplication.translate
        PLSR_Window.setWindowTitle(_translate("PLSR_Window", "PLSR_Window"))
        self.BrowsePath.setText(_translate("PLSR_Window", "Browse"))
        self.TrainData_label.setText(_translate("PLSR_Window", "Training Data Path"))
        self.ImgPth_Btn.setText(_translate("PLSR_Window", "Browse"))
        self.PathLabel.setText(_translate("PLSR_Window", "Results Directory"))
        self.attr_btn.setText(_translate("PLSR_Window", "Get attributes"))
        self.label.setText(_translate("PLSR_Window", "PLSR Regression - LOOCV"))
        self.ImgPath_label.setText(_translate("PLSR_Window", "Image Path"))
        self.SaveBtn.setText(_translate("PLSR_Window", "Save Model"))
        self.tranPath_Btn.setText(_translate("PLSR_Window", "Browse"))
        self.ReportNameLabel.setText(_translate("PLSR_Window", "Project Name"))
        self.RunBtn.setText(_translate("PLSR_Window", "Run"))

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
            self.ui = Ui_PLSR_SSS()
            self.ui.setupUi(self.window)
            return

        if not self.tranData_Path.text().endswith(".shp"):
            msg.setText("Invalid Training Data File. ")
            msg.setInformativeText("Invalid File Format")
            msg.setDetailedText(
                "The details are as follows: Either Training file is not selected or does not contain .shp extension")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_SSS()
            self.ui.setupUi(self.window)
            return

        if self.attributes.currentText() == "":
            msg.setText("Please select the attribute. ")
            msg.setInformativeText("Attribute not selected")
            msg.setDetailedText("The details are as follows: Select attribute from Training File")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_SSS()
            self.ui.setupUi(self.window)
            return

        if self.ProjectName.text() == "":
            msg.setText("Please enter a suitable name for your project. ")
            msg.setInformativeText("project name missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_SSS()
            self.ui.setupUi(self.window)
            return

        if self.savePath.text() == "":
            msg.setText("Please select path for project. ")
            msg.setInformativeText("project path missing")
            msg.exec_()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_SSS()
            self.ui.setupUi(self.window)
            return

        else:
            self.Run()

    def Run(self):
        try:
            try:
                ROOT_DIR = os.path.abspath(os.curdir)
                PROJ_DIR = ROOT_DIR + "\PROJ"
                os.environ['PROJ_LIB'] = PROJ_DIR

                if 'PROJ_LIB' in os.environ:
                    print('env variable : PROJ_LIB  is set...')
                else:
                    print('Couldnt set env variable : PROJ_LIB.Please set Manually ')
            except Exception as ex:
                print("Could not set env_var")

            log_format = "%(asctime)s::%(levelname)s::%(name)s::" \
                         "%(filename)s::%(lineno)d::%(message)s"

            logging.basicConfig(filename='app.log', filemode='w',level = logging.DEBUG,format=log_format)
            print("The process has started.... This will take few minutes")
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
                self.ui = Ui_PLSR_SSS()
                self.ui.setupUi(self.window)
                return

            else:
                os.mkdir(dir_path)

            reportpath = dir_path
            reportpath += "/" + str(self.ProjectName.text()) + "_REPORT.pdf"

            prediction_map = dir_path
            prediction_map += "/" + str(self.ProjectName.text()) + "_IMG.tif"
            if self.SaveBtn.isChecked() == True:
                modelsavepath = dir_path + "/" + self.ProjectName.text() + "_MODEL.sav"
            else:
                modelsavepath = "Model not saved"

            try:

                doc = self.rt.build_doc(reportpath, "PLSR Sparse Regression")

            except Exception as e:
                print(e)
                logging.debug("exception occurred "+e, exc_info=True)

            logging.debug("exception occurred", exc_info=True)

            loading_images = self.input_C.load_image_data()
            # print(LoadingImages)
            # print("Loading image ok")

            img_ds = loading_images[0]
            img = loading_images[1]

            print(self.attributes.currentText())
            self.input_C.set_training_attr(self.attributes.currentText())
            logging.debug("exception occurred", exc_info=True)

            # print(self.attributes.currentText() +" xxx OK")
            train_data = self.input_C.load_train_data(img_ds, "Regression")  ## roi

            logging.debug("Exception occurred", exc_info=True)
            X = img[train_data > 0, :]
            y = train_data[train_data > 0]
            logging.debug("Exception occurred", exc_info=True)

            features = pd.DataFrame(X)
            logging.debug("Exception occurred", exc_info=True)



            band_names = []
            for i in range(X.shape[1]):
                # for i in range(0,2500):
                nband = "Band_" + str(i + 1)
                band_names.append(nband)
            features.columns = band_names

            print('The shape of our features is:', features.shape)
            print('The number of Spectra is:', features.shape[0])
            print('The number of bands is:', features.shape[1])

            # print("........................ ok here ...................")
            features['value'] = y

            features.head()

            logging.debug("Exception occurred", exc_info=True)

            # Labels are the values we want to predict
            labels = np.array(features['value'])


            features = features.drop('value', axis=1)

            # Saving feature names for later use
            feature_list = list(features.columns)

            # Convert to numpy array
            features = np.array(features)

            print('Training Features Shape: ', features.shape)
            print('Training Labels Shape: ', labels.shape)


            try:
                pls_opt,mse,msemin,component,y,y_c,y_cv,score_c,score_cv = self.PLSR_C.PLSR_Regressor(features, X, y)
                # print(traceback.print_exc())

            except Exception as e:
                print(traceback.print_exc())
                logging.debug("exception occurred", exc_info=True)
                logging.debug(traceback.print_exc())
                print("regressor error")
                print(e)

            logging.exception("Exception occurred", exc_info=True)
            #print("......ok ....... regressor done")

            importance = self.PLSR_C.PLSR_vip(pls_opt)
            logging.error("Exception occurred", exc_info=True)
            cols, rows, doc, prediction , prediction_ = self.PLSR_C.PLSR_Predict(doc, score_cv, importance, img, y_c, y, y_cv,
                                                                    labels, pls_opt)
            logging.error("Exception occurred", exc_info=True)
            importance = self.PLSR_C.PLSR_vip(pls_opt)
            logging.error("Exception occurred", exc_info=True)

            self.PLSR_C.PLSR_SaveImage(prediction_map,img,img_ds,prediction)
            logging.error("Exception occurred", exc_info=True)

            if self.SaveBtn.isChecked() == True:
                self.PLSR_C.PLSR_SaveModel(pls_opt,modelsavepath)
                logging.error("Exception occurred", exc_info=True)

            rp_param = PLSR_SSS_Helper(train_data,X,mse,msemin,component,y,y_c,y_cv,self.attributes.currentText(),importance,
                                    prediction,dir_path,reportpath,prediction_map,modelsavepath,self.ImgPath.text(),self.tranData_Path.text())

            doc = self.rt.make_plsr_sss_report(doc,img,dir_path,rp_param)

            logging.error("Exception occurred", exc_info=True)
            doc.save()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Complete!!!")
            msg.setText("Processing Done. Report ready to preview ")
            msg.exec_()

        except TypeError as e:
            logging.exception("Exception occurred", exc_info=True)
            print(e)
            sys.exit(0)
