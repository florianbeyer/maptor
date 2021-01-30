# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'loadmodel.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_LoadModelWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(543, 414)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ModelPath = QtWidgets.QTextEdit(self.centralwidget)
        self.ModelPath.setGeometry(QtCore.QRect(20, 90, 311, 41))
        self.ModelPath.setObjectName("ModelPath")
        self.Browse_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.Browse_Btn.setGeometry(QtCore.QRect(380, 90, 93, 41))
        self.Browse_Btn.setObjectName("Browse_Btn")
        self.Run_Btn = QtWidgets.QPushButton(self.centralwidget)
        self.Run_Btn.setGeometry(QtCore.QRect(380, 220, 93, 41))
        self.Run_Btn.setObjectName("Run_Btn")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 543, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.Browse_Btn.clicked.connect(self.openModlePath)
        self.Run_Btn.clicked.connect(self.Execute)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Browse_Btn.setText(_translate("MainWindow", "Browse"))
        self.Run_Btn.setText(_translate("MainWindow", "Run"))

    def openModlePath(self):
        filename = QFileDialog.getOpenFileName()
        self.ModelPath.setText(filename[0])

    def Execute(self):
        pass


