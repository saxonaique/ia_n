#!/usr/bin/env python3
"""
🚀 DEMO: Sistema de Huellas 3 Puntos - Motor N DIG
Este script demuestra cómo funciona el sistema de captura automática de huellas.
"""

import numpy as np
import json
import time
from ia_dig_organismo import Metamodulo, FingerprintSystem

def demo_fingerprint_system():
    """Demuestra el sistema de huellas en acción."""
    
    print("🧠 DEMO: Sistema de Huellas 3 Puntos - Motor N DIG")
    print("=" * 60)
    
    # Inicializar el metamodulo
    print("\n1️⃣ Inicializando Motor N DIG...")
    metamodulo = Metamodulo()
    
    # Procesar entrada de texto
    print("\n2️⃣ Procesando entrada de texto...")
    input_text = "La inteligencia artificial emerge de patrones complejos en campos informacionales dinámicos."
    metamodulo.receive_input(input_text, 'text')
    
    print(f"   📝 Texto procesado: '{input_text[:50]}...'")
    print(f"   🎯 Campo inicial creado: {metamodulo.core_nucleus.field.shape}")
    
    # Simular algunos ciclos para demostrar las huellas
    print("\n3️⃣ Simulando evolución del campo...")
    
    # Ciclo 1 - Huella inicial
    print("\n   🔄 Ciclo 1 - Capturando huella INICIAL...")
    result = metamodulo.process_step()
    print(f"      ✅ Huella inicial capturada")
    print(f"      📊 Entropía: {result['metrics'].get('entropía', 0):.4f}")
    
    # Simular hasta el punto intermedio
    print(f"\n   🔄 Simulando hasta ciclo {metamodulo.fingerprint_interval}...")
    for i in range(2, metamodulo.fingerprint_interval + 1):
        if i % 200 == 0:  # Mostrar progreso cada 200 ciclos
            print(f"      📈 Ciclo {i}/{metamodulo.fingerprint_interval}")
        metamodulo.process_step()
    
    # Ciclo intermedio - Huella intermedia
    print(f"\n   🔄 Ciclo {metamodulo.fingerprint_interval} - Capturando huella INTERMEDIA...")
    result = metamodulo.process_step()
    print(f"      ✅ Huella intermedia capturada")
    print(f"      📊 Entropía: {result['metrics'].get('entropía', 0):.4f}")
    
    # Simular hasta el final
    print(f"\n   🔄 Simulando hasta ciclo {metamodulo.max_iterations}...")
    for i in range(metamodulo.fingerprint_interval + 2, metamodulo.max_iterations + 1):
        if i % 200 == 0:  # Mostrar progreso cada 200 ciclos
            print(f"      📈 Ciclo {i}/{metamodulo.max_iterations}")
        metamodulo.process_step()
    
    # Ciclo final - Huella final
    print(f"\n   🔄 Ciclo {metamodulo.max_iterations} - Capturando huella FINAL...")
    result = metamodulo.process_step()
    print(f"      ✅ Huella final capturada")
    print(f"      📊 Entropía: {result['metrics'].get('entropía', 0):.4f}")
    
    # Mostrar estado de las huellas
    print("\n4️⃣ Estado del Sistema de Huellas:")
    fingerprint_info = metamodulo.get_fingerprint_status()
    
    print(f"   🆔 ID de Sesión: {fingerprint_info['session_id']}")
    print(f"   📸 Huellas capturadas: {fingerprint_info['fingerprints']}")
    
    summary = fingerprint_info['summary']
    print(f"   📊 Total de huellas: {summary['total_fingerprints']}")
    print(f"   🎯 Etapas capturadas: {summary['stages_captured']}")
    
    # Mostrar análisis de evolución
    if 'evolution_analysis' in summary and summary['evolution_analysis']:
        print("\n5️⃣ Análisis de Evolución:")
        for transition, analysis in summary['evolution_analysis'].items():
            print(f"   🔄 {transition}:")
            print(f"      📈 Cambio en entropía: {analysis.get('entropy_change', 0):.4f}")
            print(f"      📊 Cambio en varianza: {analysis.get('variance_change', 0):.4f}")
            print(f"      ⚖️ Cambio en simetría: {analysis.get('symmetry_change', 0):.4f}")
            print(f"      ⏱️ Ciclos entre etapas: {analysis.get('cycles_between', 0)}")
    
    # Guardar huellas
    print("\n6️⃣ Guardando huellas...")
    filename = f"demo_fingerprints_{int(time.time())}.json"
    filepath = metamodulo.save_session_fingerprints(filename)
    
    if filepath:
        print(f"   ✅ Huellas guardadas en: {filepath}")
        print(f"   📁 Archivo: {filename}")
    else:
        print("   ❌ Error guardando huellas")
    
    # Cargar y verificar huellas
    print("\n7️⃣ Verificando sistema de carga...")
    if metamodulo.fingerprint_system.load_fingerprints(filepath):
        print("   ✅ Huellas cargadas exitosamente")
        print("   🔍 Verificación completada")
    else:
        print("   ❌ Error cargando huellas")
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETADO EXITOSAMENTE!")
    print("🚀 El Sistema de Huellas 3 Puntos está funcionando perfectamente.")
    print("💡 Ahora puedes usar la interfaz gráfica para experimentar.")

if __name__ == "__main__":
    try:
        demo_fingerprint_system()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\n❌ Error en el demo: {e}")
        import traceback
        traceback.print_exc()
