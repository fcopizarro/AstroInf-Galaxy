import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import requests
import io

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QPushButton, QLineEdit, QMessageBox, QTableView, QTableWidget, QTableWidgetItem   
from PyQt5.QtGui import QPixmap, QColor, QFont
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant

import skimage.draw
import urllib

# CARGA DE MODELO

ROOT_DIR = os.path.abspath("")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)  

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_images
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import galaxia

# Definir MODELO

MODEL_DIR = os.path.join(ROOT_DIR, "modelo")
WEIGHTS_PATH = os.path.join(MODEL_DIR,"galaxia_all_final.h5")

config = galaxia.GalaxiaConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2048
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

config = InferenceConfig()

class_names = ['BG',"S","E"]


model = modellib.MaskRCNN(mode="inference",config=config,model_dir=MODEL_DIR)
model.load_weights(WEIGHTS_PATH, by_name=True)


# GUI 

def get_image_sloan(_ra,_dec,_scale=0.360115,_width=512,_height=512):
    url = "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?ra="+_ra+"&dec="+_dec+"&scale="+str(_scale)+"&width="+str(_width)+"&height="+str(_height)+""
    image=skimage.io.imread(url)
    return url, image


def get_radec_from_px(x, y, ra, dec, _scale=0.360115, width=512, height=512):
    # ra, dec en el centro de la imagen
    
    # Transformar arcsec/px a grado (unidades de radec)
    deg = _scale / 3600
    
    dx = x - (width/2)
    dy = y - (height/2)
    
    # Se suman unidades hacia la izquierda, hacia arriba
    ra, dec = ra-dx*deg, dec-dy*deg
    
    return ra, dec

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None



class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()



        # Configurar la ventana principal

        self.setWindowTitle("Descartador de Galaxias")

        self.setGeometry(100, 100, 1200, 700)



        # Crear el widget principal

        main_widget = QWidget(self)
        main_widget.setFixedSize(1200, 700)

        self.setCentralWidget(main_widget)



        # Configurar el diseño principal

        main_layout = QVBoxLayout()

        main_widget.setLayout(main_layout)



        # Crear el contenedor para las ventanas gris y azul

        container_widget = QWidget(self)

        main_layout.addWidget(container_widget)



        # Configurar el diseño del contenedor

        container_layout = QHBoxLayout()

        container_widget.setLayout(container_layout)



        # Crear la ventana gris

        gray_widget = QWidget(self)

        gray_widget.setStyleSheet("background-color: #C9CBD3;")

        container_layout.addWidget(gray_widget, 1)



        # Configurar el diseño de la ventana gris

        gray_layout = QVBoxLayout(gray_widget)

        gray_layout.setAlignment(Qt.AlignTop)



        # Agregar el texto "APP" en la parte superior de la ventana gris

        app_label = QLabel("APP", self)

        app_label.setAlignment(Qt.AlignCenter)

        app_label.setFont(QFont("Tahoma", 16, QFont.Bold))

        gray_layout.addWidget(app_label)


        # # Crear el widget de pestañas

        # tab_widget = QTabWidget(self)

        # gray_layout.addWidget(tab_widget)



        # # Crear las pestañas "Imagen", "Catálogo" y "Resultado"

        # image_tab = QWidget(self)

        # catalog_tab = QWidget(self)

        # result_tab = QWidget(self)



        # # Agregar las pestañas al widget de pestañas

        # tab_widget.addTab(image_tab, "Imagen")

        # tab_widget.addTab(catalog_tab, "Catálogo")

        # tab_widget.addTab(result_tab, "Resultado")



        # # Configurar el diseño de las pestañas

        # tab_layout = QVBoxLayout()

        # image_tab.setLayout(tab_layout)



        # Agregar los botones para cambiar Imagen, Catologo, Resultados

        self.imagen_button = QPushButton('Imagen', self)
        self.imagen_button.clicked.connect(self.imagen_clicked)
        self.imagen_button.setFixedSize(200, 50)
        self.imagen_button.setStyleSheet("""
        QPushButton {
            background-color: white; 
            border: 0;
            font-weight: bold;
        }
        """)
        gray_layout.addWidget(self.imagen_button)

        self.catalogo_button = QPushButton('Catologo', self)
        self.catalogo_button.clicked.connect(self.catalogo_clicked)
        self.catalogo_button.setFixedSize(200, 50)
        self.catalogo_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)           
        gray_layout.addWidget(self.catalogo_button)

        self.resultados_button = QPushButton('Resultados', self)
        self.resultados_button.clicked.connect(self.resultados_clicked)
        self.resultados_button.setFixedSize(200, 50)
        self.resultados_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)          
        gray_layout.addWidget(self.resultados_button)



        # Espacio


    



        # Agregar las formas de cápsula y los textos "RA:" y "Dec:"

        ra_label = QLabel("RA:", self)

        ra_label.setFont(QFont("Tahoma", 12, QFont.Bold))

        gray_layout.addWidget(ra_label)

        #tab_layout.addWidget(ra_label)



        # Agregar la forma de cápsula para la RA

        self.ra_input = QLineEdit(self)
        self.ra_input.setStyleSheet("QLineEdit{ background-color: white; border: 10px solid white; border-radius: 15px;}")
        self.ra_input.resize(100, 32)

        gray_layout.addWidget(self.ra_input)

        dec_label = QLabel("Dec:", self)

        dec_label.setFont(QFont("Tahoma", 12, QFont.Bold))

        gray_layout.addWidget(dec_label)

        # Agregar la forma de cápsula para la Dec

        self.dec_input = QLineEdit(self)
        self.dec_input.setStyleSheet("QLineEdit{ background-color: white; border: 10px solid white; border-radius: 15px;}")
        self.dec_input.resize(100, 32)
        gray_layout.addWidget(self.dec_input)

        # Agregar la forma de capsula para la Escala

        esc_label = QLabel("Escala(\"/px):", self)
        esc_label.setFont(QFont("Tahoma", 12, QFont.Bold))
        gray_layout.addWidget(esc_label)

        # Agregar la forma de cápsula para la Dec

        self.esc_input = QLineEdit(self)
        self.esc_input.setStyleSheet("QLineEdit{ background-color: white; border: 10px solid white; border-radius: 15px;}")
        self.esc_input.resize(100, 32)
        gray_layout.addWidget(self.esc_input)


        # Boton para actualizar informacion
        gray_layout.addStretch(1)

        pybutton = QPushButton('Actualizar información', self)
        pybutton.clicked.connect(self.clickMethod)
        pybutton.resize(200,32)
        gray_layout.addWidget(pybutton)


        # Crear la ventana azul

        self.blue_widget = QWidget(self)

        self.blue_widget.setStyleSheet("background-color: #06134A;")

        container_layout.addWidget(self.blue_widget, 6)



        # Configurar el diseño de la ventana azul

        self.blue_layout = QHBoxLayout(self.blue_widget)


        

        # IMAGEN

        image_widget = QWidget(self)

        self.image_label = QLabel( self)
        self.pixmap = QPixmap('holder.png')
        self.image_label.setPixmap(self.pixmap)
        self.image_label.resize(512, 512)
        self.image_label.mousePressEvent = self.getPixel
        image_layout = QVBoxLayout(image_widget)
        image_layout.addWidget(self.image_label)
        self.blue_layout.addWidget(self.image_label, alignment=Qt.AlignLeft)


        


        # Agregar el cuadrado blanco con el texto "Seleccionado"

        selected_widget = QWidget(self)

        selected_widget.setFixedSize(300, 350)

        selected_widget.setStyleSheet("background-color: white; border-radius: 20px;")

        selected_layout = QVBoxLayout(selected_widget)

        selected_layout.setAlignment(Qt.AlignCenter)

        selected_label = QLabel("Seleccionado:", self)

        selected_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        pixel_label = QLabel("Pixel:", self)

        self.x_label = QLabel("", self)
        self.y_label = QLabel("", self)

        coord_label = QLabel("Coord (Ra-Dec):", self)

        self.ra_x_label = QLabel("", self)
        self.dec_y_label = QLabel("", self)

        selected_label.setFont(QFont("Tahoma", 12, QFont.Bold))

        pixel_label.setFont(QFont("Tahoma", 12, QFont.Bold))

        coord_label.setFont(QFont("Tahoma", 12, QFont.Bold))

        selected_layout.addWidget(selected_label)

        selected_layout.addWidget(pixel_label)
        selected_layout.addWidget(self.x_label)
        selected_layout.addWidget(self.y_label)

        selected_layout.addWidget(coord_label)
        selected_layout.addWidget(self.ra_x_label)
        selected_layout.addWidget(self.dec_y_label)

        
        # Agregar un espacio flexible antes del botón

        selected_layout.addStretch()



        # Agregar el botón "Detectar galaxias" dentro del cuadrado blanco

        self.n_galaxies = QLabel("", self)

        detect_button = QPushButton("Detectar galaxias", self)

        detect_button.clicked.connect(self.detectGalaxies)

        detect_button.setStyleSheet("background-color: #06134A; color: white;")

        detect_button.setFixedSize(250, 50)     

        font = QFont("Tahoma")

        font.setWeight(QFont.Bold)

        detect_button.setFont(font)

        selected_layout.addWidget(self.n_galaxies)
        selected_layout.addWidget(detect_button)

        self.blue_layout.addWidget(selected_widget, alignment=Qt.AlignRight)

        
        # -------------------- CATALOGO -----------------------------------

        self.catalogo = None

        self.catalogo_widget = QWidget(self)
        self.catalogo_widget.setStyleSheet("background-color: #C9CBD3;")
        self.catalogo_widget.setFixedSize(940, 600)
        self.catalogo_widget.hide()

        self.catalogo_layout = QHBoxLayout(self.catalogo_widget)
        container_layout.addWidget(self.catalogo_widget, 6)

        # ----------------------- RESULTADO -------------------------------------

        self.resultado = None

        self.resultado_widget = QWidget(self)
        self.resultado_widget.setStyleSheet("background-color: #C9CBD3;")
        self.resultado_widget.setFixedSize(940, 600)
        self.resultado_widget.hide()

        self.resultado_layout = QHBoxLayout(self.resultado_widget)
        container_layout.addWidget(self.resultado_widget, 6)











    def getPixel(self , event):
        x = event.pos().x()
        y = event.pos().y() - 66
        print(x,y)
        self.x_label.setText(str(x))
        self.y_label.setText(str(y))
        ra_dec = get_radec_from_px(
                    x,
                    y, 
                    float(self.ra_input.text()), 
                    float(self.dec_input.text()))
        self.ra_x_label.setText(str(ra_dec[0]))
        self.dec_y_label.setText(str(ra_dec[1]))




    def clickMethod(self):
        url_skyserver, self.image = get_image_sloan(self.ra_input.text(), self.dec_input.text(), self.esc_input.text())
        with urllib.request.urlopen(url_skyserver) as url:
            data = url.read()
        self.pixmap.loadFromData(data)
        self.image_label.setPixmap(self.pixmap) 

        url_catalogo = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/RadialSearch?ra="+self.ra_input.text()+"&dec="+self.dec_input.text()+"&radius=3&whichway=equatorial&limit=20&format=csv&fp=none&whichquery=imaging"
        self.catalogo = pd.read_csv(url_catalogo, header=1)

        # Limpiar catalogo
        children = []
        for i in range(self.catalogo_layout.count()):
            child = self.catalogo_layout.itemAt(i).widget()
            if child:
                children.append(child)
        for child in children:
            child.deleteLater()

        model = pandasModel(self.catalogo)
        view = QTableView()
        view.setModel(model)
        view.setStyleSheet("QTableWidget::item {border: 0px; padding: 5px; background-color: white}")
        view.resize(940, 600)
        self.catalogo_layout.addWidget(view)


    def imagen_clicked(self):
        self.blue_widget.show()
        self.catalogo_widget.hide()
        self.resultado_widget.hide()


        self.imagen_button.setStyleSheet("""
        QPushButton {
            background-color: white; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: white; 
            }
        """)  

        self.catalogo_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)         
        
        self.resultados_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3;; 
            }
        """)        

    def catalogo_clicked(self):
        self.blue_widget.hide()
        self.catalogo_widget.show()
        self.catalogo_widget.setFixedSize(940, 600)
        self.resultado_widget.hide()


        self.imagen_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)  

        self.catalogo_button.setStyleSheet("""
        QPushButton {
            background-color: white; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: white; 
            }
        """)         
        
        self.resultados_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)        

    def resultados_clicked(self):
        self.blue_widget.hide()
        self.catalogo_widget.hide()
        self.resultado_widget.show()
        self.resultado_widget.setFixedSize(940, 600)

        self.imagen_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)  

        self.catalogo_button.setStyleSheet("""
        QPushButton {
            background-color: #C9CBD3; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #dadce3; 
            }
        """)         
        
        self.resultados_button.setStyleSheet("""
        QPushButton {
            background-color: white; 
            border: 0;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: white; 
            }
        """)          


    def detectGalaxies(self):
        results = model.detect([self.image], verbose=1)
        r = results[0]
        

        print(len(r['scores']), "Objetos detectados RA-DEC:")

        self.n_galaxies.setText("Se han detectado "+str(len(r['scores']))+" galaxias.")

        for i in r['rois']:
            print (
                i[1], i[3], i[0], i[2],
                get_radec_from_px(
                    np.mean([i[1],i[3]]),
                    np.mean([i[0],i[2]]), 
                    float(self.ra_input.text()), 
                    float(self.dec_input.text()))
            )

        # Get input and output to classifier and mask heads.
        mrcnn = model.run_graph([self.image], [
            ("proposals", model.keras_model.get_layer("ROI").output),
            ("probs", model.keras_model.get_layer("mrcnn_class").output),
            ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ])

        # Get detection class IDs. Trim zero padding.
        det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
        det_count = np.where(det_class_ids == 0)[0][0]
        det_class_ids = det_class_ids[:det_count]
        detections = mrcnn['detections'][0, :det_count]

        #captions = ["{} {:.3f}".format(class_names[int(c)], s) if c > 0 else ""
        #            for c, s in zip(detections[:, 4], detections[:, 5])]

        # Limpiar captions
        captions = [""
                    for c, s in zip(detections[:, 4], detections[:, 5])]
        visualize.draw_boxes(
            self.image, 
            refined_boxes=utils.denorm_boxes(detections[:, :4], self.image.shape[:2]),
            visibilities=[2] * len(detections),
            captions=captions, title="Detections",
            ax=get_ax())

        self.pixmap = QPixmap('temp.png')
        self.pixmap = self.pixmap.scaled(512, 512)

        self.image_label.setPixmap(self.pixmap)





        self.resultado = self.catalogo.copy()
        print(self.resultado.info())

        for i in r['rois']:

            ld_pnt = get_radec_from_px(
                    i[1], i[0],
                    float(self.ra_input.text()), 
                    float(self.dec_input.text()))
            
            lu_pnt = get_radec_from_px(
                    i[1], i[2],
                    float(self.ra_input.text()), 
                    float(self.dec_input.text()))
        
            rd_pnt = get_radec_from_px(
                    i[3], i[0],
                    float(self.ra_input.text()), 
                    float(self.dec_input.text()))
            
            ru_pnt = get_radec_from_px(
                    i[3], i[2],
                    float(self.ra_input.text()), 
                    float(self.dec_input.text()))
            
            print(ld_pnt, rd_pnt)
            print(lu_pnt, ru_pnt)

            indexAge = self.resultado[ (self.resultado['ra'] <= ld_pnt[0]) & (self.resultado['ra'] >= rd_pnt[0]) &
                                       (self.resultado['dec'] <= ld_pnt[1]) & (self.resultado['dec'] >= lu_pnt[1])
                                      ].index
            print(indexAge)
            self.resultado.drop(indexAge , inplace=True)

        # Limpiar resultado
        children = []
        for i in range(self.resultado_layout.count()):
            child = self.resultado_layout.itemAt(i).widget()
            if child:
                children.append(child)
        for child in children:
            child.deleteLater()

        model_res = pandasModel(self.resultado)
        view_res = QTableView()
        view_res.setModel(model_res)
        view_res.resize(940, 600)
        self.resultado_layout.addWidget(view_res)
            














if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec_())

