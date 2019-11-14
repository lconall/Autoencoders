from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import random

import KerasAutoencoder, MINSTDataLoader, NoisyMINSTDataLoader
from KerasAutoencoder.run_autoencoder import RunAutoencoder
from MINSTDataLoader.data_loader import MINSTDataLoader
from NoisyMINSTDataLoader.data_loader import NoisyMINSTDataLoader
 

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(186, 141)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.model_name_comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.model_name_comboBox.setGeometry(QtCore.QRect(10, 10, 161, 31))
        self.model_name_comboBox.setObjectName("model_name_comboBox")
        self.model_name_comboBox.addItem("")
        self.model_name_comboBox.addItem("")

        self.epochs_label = QtWidgets.QLabel(self.centralwidget)
        self.epochs_label.setGeometry(QtCore.QRect(50, 50, 70, 31))
        self.epochs_label.setText("# Epochs")

        self.epochs_textbox = QtWidgets.QTextEdit(self.centralwidget)
        self.epochs_textbox.setGeometry(QtCore.QRect(120, 50, 50, 31))
        self.epochs_textbox.setText("50")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 90, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.run_selected_autoencoder)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 186, 22))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.model_name_comboBox.setItemText(0, _translate("MainWindow", "Simple Autoencoder"))
        self.model_name_comboBox.setItemText(1, _translate("MainWindow", "Denoising Autoencoder"))
        self.pushButton.setText(_translate("MainWindow", "Run Example"))

    def run_selected_autoencoder(self): 
        self.centralwidget.hide
        try: 
            epochs = int(self.epochs_textbox.toPlainText())
        except Exception as err: 
            print("ERROR - epochs must be integer!")
            raise err
        model_name = self.model_name_comboBox.currentText()
        if model_name == "Simple Autoencoder":
            minst_data_loader = MINSTDataLoader()
            run_autoencoder = RunAutoencoder(data_loader=minst_data_loader)
            results = run_autoencoder.run(36, epochs=epochs)
            self.display_results(model_name, results)
        elif model_name == "Denoising Autoencoder":
            noisy_minst_data_loader = NoisyMINSTDataLoader()
            run_autoencoder = RunAutoencoder(data_loader=noisy_minst_data_loader)
            results = run_autoencoder.run(36, epochs=epochs)
            self.display_results(model_name, results)

    def display_results(self, model_name, results): 
        nrows, ncols, figsize = 4, 3, [12, 6]
        figure, all_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plt.gray()
        figure.suptitle(model_name + " Visualization (Target, Input, Encoded_Input, Reconstruction)", weight="bold")
        
        input_numbers = []
        for figure_index, axis_at_index in enumerate(all_axes.flat):
            # figure_index runs from 0 to (nrows*ncols-1)
            # axis_at_index is equivalent with ax[rowid][colid]
            
            row_id, col_id = figure_index // ncols, figure_index % ncols            
            if figure_index < ncols: 
                input_numbers.append(random.randint(0, len(results['test_target'])))

            if row_id == 0:  
                axis_at_index.imshow(results['test_target'][input_numbers[col_id]].reshape(28, 28), alpha=1)
                axis_at_index.set_title("Test Target " + str(input_numbers[col_id]))
            elif row_id == 1:
                axis_at_index.imshow(results['test_data'][input_numbers[col_id]].reshape(28, 28), alpha=1)
                axis_at_index.set_title("Test Input " + str(input_numbers[col_id]))
            elif row_id == 2:
                height, width = 6, 6
                axis_at_index.imshow(results['encoded_data'][input_numbers[col_id]].reshape(6, 6), alpha=1)
                axis_at_index.set_title("Encoded Data " + str(input_numbers[col_id]))
            elif row_id == 3:
                axis_at_index.imshow(results['decoded_data'][input_numbers[col_id]].reshape(28, 28), alpha=1)
                axis_at_index.set_title("Reconstructed " + str(input_numbers[col_id]))

        plt.tight_layout(True)
        plt.subplots_adjust(top=0.80, bottom=0.1)
        plt.show()
