import sys, os
# 🔹 Añadimos la ruta al proyecto para que pueda encontrar ia_dig_organismo.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neo import CoreNucleus


def test_coreGI(iterations: int = 50):
    """
    Test de estabilidad y métricas del CoreNucleus con integración de Gi.
    """
    print("=== INICIO TEST CoreGI ===")
    core = CoreNucleus()
    print("[CoreNucleus] INFO: Inicializado.")

    for i in range(1, iterations + 1):
        core.iterate_field()

        metrics = core.get_metrics()
        S = metrics.get("entropia", 0.0)
        V = metrics.get("varianza", 0.0)
        M = metrics.get("maximo", 0.0)
        Sym = metrics.get("simetria", 0.0)

        print(
            f"Iter {i:02d} | S={S:.3f} | V={V:.3f} | M={M:.3f} | Sym={Sym:.3f}"
        )

    print("=== FIN TEST CoreGI ===")


if __name__ == "__main__":
    test_coreGI(iterations=50)


