try:
    import sys,logging
    from PyQt5 import QtCore, QtGui, QtWidgets
    from RF_NSS import Ui_RF_NSS
    from PLSR_SSS import Ui_PLSR_SSS
    from PLSR_LDS import Ui_PLSR_LDS
    from RFR_SSS import Ui_RFR_SSS
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

except Exception as e:
    print('Can not import files:' + str(e))
    input("Press Enter to exit!")
    sys.exit(0)


class Ui_RegressionTypes(object):
    def setupUi(self, RegressionTypes):
        RegressionTypes.setObjectName("RegressionTypes")
        RegressionTypes.resize(800, 667)
        self.centralwidget = QtWidgets.QWidget(RegressionTypes)
        self.centralwidget.setObjectName("centralwidget")
        self.Header = QtWidgets.QLabel(self.centralwidget)
        self.Header.setGeometry(QtCore.QRect(250, 10, 301, 61))
        font = QtGui.QFont()
        font.setPointSize(19)
        self.Header.setFont(font)
        self.Header.setObjectName("Header")
        self.Next_btn = QtWidgets.QPushButton(self.centralwidget)
        self.Next_btn.setGeometry(QtCore.QRect(600, 550, 161, 61))
        self.Next_btn.setObjectName("Next_btn")
        self.RF_SP_btn = QtWidgets.QRadioButton(self.centralwidget)
        self.RF_SP_btn.setGeometry(QtCore.QRect(60, 440, 131, 20))
        self.RF_SP_btn.setObjectName("RF_SP_btn")
        self.SparseSample_label = QtWidgets.QLabel(self.centralwidget)
        self.SparseSample_label.setGeometry(QtCore.QRect(20, 350, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.SparseSample_label.setFont(font)
        self.SparseSample_label.setObjectName("SparseSample_label")
        self.PLSR_SP_btn = QtWidgets.QRadioButton(self.centralwidget)
        self.PLSR_SP_btn.setGeometry(QtCore.QRect(60, 490, 95, 20))
        self.PLSR_SP_btn.setObjectName("PLSR_SP_btn")
        self.NormalSample_label_3 = QtWidgets.QLabel(self.centralwidget)
        self.NormalSample_label_3.setGeometry(QtCore.QRect(20, 380, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.NormalSample_label_3.setFont(font)
        self.NormalSample_label_3.setObjectName("NormalSample_label_3")
        self.NormalSample_label = QtWidgets.QLabel(self.centralwidget)
        self.NormalSample_label.setGeometry(QtCore.QRect(20, 130, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.NormalSample_label.setFont(font)
        self.NormalSample_label.setObjectName("NormalSample_label")
        self.NormalSample_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.NormalSample_label_2.setGeometry(QtCore.QRect(20, 160, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.NormalSample_label_2.setFont(font)
        self.NormalSample_label_2.setObjectName("NormalSample_label_2")
        self.PLSR_NN_btn = QtWidgets.QRadioButton(self.centralwidget)
        self.PLSR_NN_btn.setGeometry(QtCore.QRect(50, 270, 95, 20))
        self.PLSR_NN_btn.setObjectName("PLSR_NN_btn")
        self.RF_NN_btn = QtWidgets.QRadioButton(self.centralwidget)
        self.RF_NN_btn.setGeometry(QtCore.QRect(50, 210, 181, 41))
        self.RF_NN_btn.setObjectName("RF_NN_btn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(210, 270, 121, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(210, 440, 131, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        RegressionTypes.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(RegressionTypes)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        RegressionTypes.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(RegressionTypes)
        self.statusbar.setObjectName("statusbar")
        RegressionTypes.setStatusBar(self.statusbar)

        self.Next_btn.clicked.connect(self.RegressionSelection)
        self.retranslateUi(RegressionTypes)
        QtCore.QMetaObject.connectSlotsByName(RegressionTypes)

    def retranslateUi(self, RegressionTypes):
        _translate = QtCore.QCoreApplication.translate
        RegressionTypes.setWindowTitle(_translate("RegressionTypes", "RegressionTypes"))
        self.Header.setText(_translate("RegressionTypes", "Types of Regression"))
        self.Next_btn.setText(_translate("RegressionTypes", "Next"))
        self.RF_SP_btn.setText(_translate("RegressionTypes", "Random Forest"))
        self.SparseSample_label.setText(_translate("RegressionTypes", "Sparse Sample Size    (e.g. n < 50)"))
        self.PLSR_SP_btn.setText(_translate("RegressionTypes", "PLSR"))
        self.NormalSample_label_3.setText(_translate("RegressionTypes", "*Leave one out Cross Validation (LOOCV) will be used instead of spliting training data."))
        self.NormalSample_label.setText(_translate("RegressionTypes", "Normal Sample Size    (e.g. n > 150)"))
        self.NormalSample_label_2.setText(_translate("RegressionTypes", "*Sampling data will be splitted in training and test data according to user input."))
        self.PLSR_NN_btn.setText(_translate("RegressionTypes", "PLSR    "))
        self.RF_NN_btn.setText(_translate("RegressionTypes", "Random Forest"))
        # self.label.setText(_translate("RegressionTypes", "Not available yet*"))
        # self.label_2.setText(_translate("RegressionTypes", "Not available yet*"))

    def RegressionSelection(self):

        if self.RF_NN_btn.isChecked() == True:
            self.RF_SP_btn.setChecked(False)
            self.PLSR_NN_btn.setChecked(False)
            self.PLSR_SP_btn.setChecked(False)

            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RF_NSS()
            self.ui.setupUi(self.window)
            self.window.show()

        if self.PLSR_NN_btn.isChecked() == True:

            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_LDS()
            self.ui.setupUi(self.window)
            self.window.show()

        if self.RF_SP_btn.isChecked() == True:
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_RFR_SSS()
            self.ui.setupUi(self.window)
            self.window.show()

        if self.PLSR_SP_btn.isChecked() == True:
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_PLSR_SSS()
            self.ui.setupUi(self.window)
            self.window.show()
