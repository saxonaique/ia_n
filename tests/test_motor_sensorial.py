from motor_sensitivo_vsc import Campo, cargar_diccionario, inyectar_letra

if __name__ == "__main__":
    campo = Campo()
    letras = cargar_diccionario()

    for letra_data in letras:
        print(f"Inyectando letra: {letra_data['letra']}")
        inyectar_letra(campo, letra_data)

    estado = campo.exportar_estado()
    print(estado)  # <- para verificar si hay nodos
