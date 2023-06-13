# AstroInf-Galaxy

Debido a la falta de actualizacion del algoritmo de deteccion usado se debe usar una version mas antigua de python. En nuestro caso esto fue hecho creando un entorno virtual de Anaconda, con la version de Python 3.7.

Para usar aproximadamente el mismo entorno que usamos es necesario re-crear los paquetes que usa el entorno, para esto se provee, de la descripcion del entorno "py37.yml", pudiendo copiarse con:

```
conda env create -f py37.yml
```

Y, una vez instalado todos los paquetes, se debe activar el mismo:

```
conda activate py37
```


Luego, para correr el programa es necesario descargar el modelo entrenado, por problemas de espacio se debe descargar aparte:

![https://drive.google.com/file/d/1XkWkcpE5qwZ8xeelPUz7zOZfZhrK3HR-/view?usp=sharing](https://drive.google.com/file/d/1XkWkcpE5qwZ8xeelPUz7zOZfZhrK3HR-/view?usp=sharing)

Y ser puesto dentro de la carpeta /modelo/. Si todo está bien, entonces se puede iniciar la aplicacion con el archivo "app_astroinf.py":

```
python app_astroinf.py
```
