
import json

def cargar_tarjeta(ruta):
    with open(ruta, "r") as archivo:
        tarjeta = json.load(archivo)
    return tarjeta
