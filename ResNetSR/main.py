from PyQt5.QtCore import QFile
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap

def main():
    app = QApplication([])
    window = uic.loadUi("main.ui")

    style = QFile("style.qss")
    style.open(QFile.ReadOnly | QFile.Text)
    app.setStyleSheet(bytes(style.readAll()).decode())

    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(20)
    shadow.setXOffset(-10)
    shadow.setYOffset(10)
    shadow.setColor(QColor(0, 0, 0, 50))

    # Set up the main layout
    main_layout = QVBoxLayout(window.centralwidget)
    main_layout.setObjectName("main_layout")

    # Set up the image input and output labels
    input_layout = QHBoxLayout()
    input_layout.setObjectName("image_layout")
    input_label = QLabel()
    input_label.setObjectName("input_label")
    input_label.setScaledContents(True)
    input_layout.addWidget(input_label)
    main_layout.addLayout(input_layout)
    output_label = QLabel()
    output_label.setObjectName("output_label")
    output_label.setScaledContents(True)
    input_layout.addWidget(output_label)

    # Set up the button layout
    button_layout = QHBoxLayout()
    button_layout.setObjectName("button_layout")
    imp_img_btn = QPushButton("Import")
    imp_img_btn.setObjectName("imp_img_btn")
    button_layout.addWidget(imp_img_btn)
    run_btn = QPushButton("Run ResNetSR")
    run_btn.setObjectName("run_btn")
    button_layout.addWidget(run_btn)
    reset_btn = QPushButton("Reset")
    reset_btn.setObjectName("reset_btn")
    button_layout.addWidget(reset_btn)
    main_layout.addLayout(button_layout)

    # Set the main layout as the central widget's layout
    window.centralwidget.setLayout(main_layout)

    # Connect the buttons to their respective functions
    run_btn.clicked.connect(lambda: onClick(window))
    imp_img_btn.clicked.connect(lambda: findImg(window))
    reset_btn.clicked.connect(lambda: resetState(window))

    window.show()
    app.exec()

def findImg(window):
    fname, _ = QFileDialog.getOpenFileName(window, "Open File", "C:\\Users\\Mohamad\\Pictures", "PNG Files (*.png);;Jpg files (*.jpg)")
    pixmap = QPixmap(fname)
    input_label = window.findChild(QLabel, "input_label")  # get the input label object by name
    input_label.setPixmap(pixmap)
    window.input_path = fname  # assign the input file path to the window object


completed = False

def onClick(window):
    from run import process_image
    # Get the path to the input image
    input_path = window.input_path

    # Process the input image
    output_path = process_image(input_path)

    # Load the output image as a QPixmap
    output_pixmap = QPixmap(output_path)

    output_label = window.findChild(QLabel, "output_label")  # get the output label object by name
    # Set the output label pixmap to the processed image
    output_label.setPixmap(output_pixmap)

    # Wait for the process to complete
    while not completed:
        QApplication.processEvents()

    # Update the UI once the process is complete
    run_btn = window.findChild(QPushButton, "run_btn")  # get the run button object by name
    run_btn.setEnabled(True)
    lbl_status = window.findChild(QLabel, "lbl_status")  # get the label object by name
    lbl_status.setText("Completed")

def resetState(window):
    # Reset input and output labels to their initial state
    input_label = window.findChild(QLabel, "input_label")  # get the input label object by name
    input_label.setPixmap(QPixmap())
    output_label = window.findChild(QLabel, "output_label")  # get the output label object by name
    output_label.setPixmap(QPixmap())

if __name__ == '__main__':
    main()
