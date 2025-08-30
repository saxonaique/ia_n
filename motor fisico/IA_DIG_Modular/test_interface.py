#!/usr/bin/env python3
"""
🧪 TEST: Verificación de la Interfaz de Aprendizaje
Este script verifica que todos los elementos de la interfaz se muestren correctamente.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Agregar el directorio actual al path para importar ia_dig_organismo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_interface():
    """Prueba la interfaz paso a paso."""
    
    print("🧪 Iniciando prueba de interfaz...")
    
    try:
        # Importar la aplicación
        from neo import DIGVisualizerApp
        
        print("✅ Importación exitosa")
        
        # Crear la aplicación
        print("🔧 Creando aplicación...")
        app = DIGVisualizerApp()
        
        print("✅ Aplicación creada")
        
        # Verificar que los elementos estén presentes
        print("🔍 Verificando elementos de la interfaz...")
        
        # Verificar panel de aprendizaje
        if hasattr(app, 'learning_status_label'):
            print("✅ learning_status_label encontrado")
        else:
            print("❌ learning_status_label NO encontrado")
        
        if hasattr(app, 'custom_label_entry'):
            print("✅ custom_label_entry encontrado")
        else:
            print("❌ custom_label_entry NO encontrado")
        
        if hasattr(app, 'learn_button'):
            print("✅ learn_button encontrado")
        else:
            print("❌ learn_button NO encontrado")
        
        if hasattr(app, 'recognize_button'):
            print("✅ recognize_button encontrado")
        else:
            print("❌ recognize_button NO encontrado")
        
        # Verificar panel de huellas
        if hasattr(app, 'fingerprint_status_label'):
            print("✅ fingerprint_status_label encontrado")
        else:
            print("❌ fingerprint_status_label NO encontrado")
        
        # Mostrar la aplicación por 5 segundos
        print("🖥️ Mostrando interfaz por 5 segundos...")
        app.after(5000, app.quit)
        app.mainloop()
        
        print("✅ Prueba completada exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interface()
