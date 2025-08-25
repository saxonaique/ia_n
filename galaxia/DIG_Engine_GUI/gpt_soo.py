
def analizar_tarjeta(tarjeta):
    tema = tarjeta.get("tema", "desconocido")
    variables = tarjeta.get("variables", [])
    analisis = f"Tarjeta de tipo '{tema}' con variables: {', '.join(variables)}"
    return analisis
