from PyQt5 import QtCore, QtGui, QtWidgets
from ClassificationWindow import Ui_ClassificationWindow
from loadmodel import Ui_LoadModelWindow
from RegressionWindow import Ui_RegressionWindow

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(563, 767)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Classification_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Classification_btn.setGeometry(QtCore.QRect(170, 170, 191, 71))
        self.Classification_btn.setObjectName("Classification_btn")
        self.Regression_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Regression_btn.setGeometry(QtCore.QRect(170, 270, 191, 71))
        self.Regression_btn.setObjectName("Regression_btn")
        self.Help_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Help_btn.setGeometry(QtCore.QRect(170, 470, 191, 71))
        self.Help_btn.setObjectName("Help_btn")
        self.Exit_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Exit_btn.setGeometry(QtCore.QRect(170, 570, 191, 71))
        self.Exit_btn.setObjectName("Exit_btn")
        self.LoadModel_btn = QtWidgets.QPushButton(self.centralwidget)
        self.LoadModel_btn.setGeometry(QtCore.QRect(170, 370, 191, 71))
        self.LoadModel_btn.setObjectName("LoadModel_btn")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(430, 0, 121, 91))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("C:/Users/User/Downloads/rsz_rsz_uni_rostock_logo-removebg-preview.png"))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(220, 680, 361, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 0, 451, 161))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("C:/Users/User/Downloads/LogoMakr_56w21a.png"))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 563, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.Classification_btn.clicked.connect(self.OpenClassificationWindow)
        self.Regression_btn.clicked.connect(self.OpenrRegressionnWindow)
        self.LoadModel_btn.clicked.connect(self.OpenLoadModelWindow)

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
        self.label_3.setText(_translate("MainWindow", "Version 1.1"))


    def OpenClassificationWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_ClassificationWindow()
        self.ui.setupUi(self.window)
        # MainWindow.hide()
        self.window.show()

    def OpenLoadModelWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_LoadModelWindow()
        self.ui.setupUi(self.window)

    # MainWindow.hide()
    # self.window.show()

    def OpenrRegressionnWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_RegressionWindow()
        self.ui.setupUi(self.window)
        # MainWindow.hide()
        self.window.show()

if __name__ == "__main__":
    try:
        import sys
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
    except:
        print("Could not load Application on System.")