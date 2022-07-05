import gym
import grid_world
from q_learning_agent import QLearningAgent
import numpy as np
import random
from PyQt6 import QtCore, QtGui, QtWidgets


import interfaceTest
from interfaceTest import Ui_Form

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec())





