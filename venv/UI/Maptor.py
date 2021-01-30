# try:
import sys,logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from Classification import Ui_ClassificationWindow
from loadmodel import Ui_LoadModelWindow
from HelpWindow import Ui_HelpWindow
from RegressionTypes import Ui_RegressionTypes
# except Exception as e:
#     logging.error("Exception occurred", exc_info=True)
#     print('Can not import files:' + str(e))
#     input("Press Enter to exit!")
#     sys.exit(0)



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(701, 852)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Classification_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Classification_btn.setGeometry(QtCore.QRect(80, 210, 221, 91))
        self.Classification_btn.setObjectName("Classification_btn")
        self.Regression_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Regression_btn.setGeometry(QtCore.QRect(80, 330, 221, 91))
        self.Regression_btn.setObjectName("Regression_btn")
        self.Help_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Help_btn.setGeometry(QtCore.QRect(410, 330, 221, 91))
        self.Help_btn.setObjectName("Help_btn")
        self.Exit_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Exit_btn.setGeometry(QtCore.QRect(240, 470, 221, 91))
        self.Exit_btn.setObjectName("Exit_btn")
        self.LoadModel_btn = QtWidgets.QPushButton(self.centralwidget)
        self.LoadModel_btn.setGeometry(QtCore.QRect(410, 210, 221, 91))
        self.LoadModel_btn.setObjectName("LoadModel_btn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(290, 660, 141, 121))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(r"Images/rsz_11rsz_euro_fonds_quer.png"))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(160, -50, 431, 241))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(r"Images/Mapto.png"))
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(310, 620, 141, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 650, 141, 121))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(r"Images/unirostock.png"))
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(160, 675, 101, 81))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap(r"Images/rsz_1rsz_1europaeischer-sozialfonds-vektor-farbig-rgb.png"))
        self.label_6.setObjectName("label_6")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(430, 670, 231, 91))
        self.label_5.setMinimumSize(QtCore.QSize(0, 0))
        self.label_5.setMaximumSize(QtCore.QSize(1000, 1000))
        self.label_5.setSizeIncrement(QtCore.QSize(1000, 1000))
        self.label_5.setBaseSize(QtCore.QSize(1000, 1000))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap(r"Images/rsz_1wetscapes_logo.png"))
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(50, 740, 621, 91))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 701, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.Classification_btn.clicked.connect(self.OpenClassificationWindow)
        self.Regression_btn.clicked.connect(self.OpenRegressionnWindow)
        self.LoadModel_btn.clicked.connect(self.OpenLoadModelWindow)
        self.Help_btn.clicked.connect(self.OpenHelpWindow)
        self.Exit_btn.clicked.connect(self.CloseWindow)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Maptor"))
        self.Classification_btn.setText(_translate("MainWindow", "Classification"))
        self.Regression_btn.setText(_translate("MainWindow", "Regression"))
        self.Help_btn.setText(_translate("MainWindow", "Help"))
        self.Exit_btn.setText(_translate("MainWindow", "Exit"))
        self.LoadModel_btn.setText(_translate("MainWindow", "Load Model"))
        self.label_3.setText(_translate("MainWindow", "Version 1.4 "))
        self.label_7.setText(_translate("MainWindow", "The European Social Fund (ESF) and the Ministry of Education, Science and Culture of Mecklenburg-Western Pomerania funded this work within the project WETSCAPES \n \t\t\t\t\t\t\t\t\t\t\t(ESF/14-BM-A55-0028/16)."))

    def OpenClassificationWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_ClassificationWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def OpenLoadModelWindow(self):

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Work in Progress")
        msg.setText("This feature will be available in next releases. ")
        msg.exec_()
        return

    def OpenRegressionnWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_RegressionTypes()
        self.ui.setupUi(self.window)
        self.window.show()

    def OpenHelpWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_HelpWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def CloseWindow(self):
        sys.exit()


if __name__ == "__main__":
    try:
        import sys,os
        try:

            if 'PROJ_LIB' in os.environ:
                del os.environ["PROJ_LIB"]

            ROOT_DIR = os.path.abspath(os.curdir)
            PROJ_DIR = ROOT_DIR + "\PROJ"
            os.environ['PROJ_LIB'] = PROJ_DIR

            if 'PROJ_LIB' in os.environ:
                print('env variable : PROJ_LIB  set...')
            else:
                print('Couldnt set env variable : PROJ_LIB.Please set Manually ')


            for i, j in os.environ.items():
                print(i, j)

        except Exception as ex:
            print("Could not set env_var")

        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
        print("Could not load Application on System.")