#!/usr/bin/env python3
"""
🧠 DEMO: Sistema de Aprendizaje por Reconocimiento de Patrones - Motor N DIG
Este script demuestra cómo la IA aprende y reconoce patrones repetidos.
"""

import time
from neo import Metamodulo

def demo_learning_system():
    """Demuestra el sistema de aprendizaje en acción."""
    
    print("🧠 DEMO: Sistema de Aprendizaje por Reconocimiento de Patrones")
    print("=" * 70)
    
    # Inicializar el metamodulo
    print("\n1️⃣ Inicializando Motor N DIG con Sistema de Aprendizaje...")
    metamodulo = Metamodulo()
    
    # Lista de textos para probar el aprendizaje
    test_texts = [
        "La inteligencia artificial emerge de patrones complejos en campos informacionales dinámicos.",
        "Los sistemas neuronales procesan información a través de conexiones sinápticas adaptativas.",
        "La gravedad emerge de un balance dinámico entre desorden (S) y orden (I).",
        "Los fractales representan la belleza matemática de la naturaleza caótica.",
        "La inteligencia artificial emerge de patrones complejos en campos informacionales dinámicos.",  # REPETIDO
        "Los sistemas neuronales procesan información a través de conexiones sinápticas adaptativas.",  # REPETIDO
        "La evolución biológica sigue patrones de selección natural y mutación genética.",
        "La inteligencia artificial emerge de patrones complejos en campos informacionales dinámicos.",  # REPETIDO 2
    ]
    
    print(f"\n2️⃣ Probando con {len(test_texts)} textos diferentes...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n   📝 Texto {i}: '{text[:50]}...'")
        
        # Procesar el texto
        metamodulo.receive_input(text, 'text')
        
        # Simular algunos ciclos para generar huellas
        print(f"      🔄 Simulando evolución...")
        for cycle in range(1, 1001):  # Solo hasta 1000 para acelerar
            if cycle % 200 == 0:
                print(f"         📈 Ciclo {cycle}/1000")
            metamodulo.process_step()
        
        # Verificar si es un patrón repetido
        print(f"      🔍 Verificando reconocimiento...")
        recognized_pattern = metamodulo.recognize_pattern(text, 'text')
        
        if recognized_pattern:
            label = recognized_pattern.get('auto_label', 'Desconocido')
            similarity = recognized_pattern.get('similarity_score', 0.0)
            usage_count = recognized_pattern.get('usage_count', 0)
            
            print(f"         ✅ RECONOCIDO: {label}")
            print(f"            📊 Similitud: {similarity:.3f}")
            print(f"            🔢 Veces usado: {usage_count}")
        else:
            print(f"         ❓ NO RECONOCIDO - Patrón nuevo")
            
            # Aprender el nuevo patrón
            print(f"         🎓 Aprendiendo nuevo patrón...")
            pattern_id = metamodulo.learn_from_session(text, 'text')
            
            if pattern_id:
                print(f"            ✅ Aprendido: {pattern_id}")
            else:
                print(f"            ❌ Error aprendiendo")
        
        # Pausa entre textos
        time.sleep(0.5)
    
    # Mostrar estadísticas finales
    print("\n3️⃣ Estadísticas del Sistema de Aprendizaje:")
    learning_stats = metamodulo.get_learning_stats()
    
    print(f"   📚 Total de patrones aprendidos: {learning_stats.get('total_patterns', 0)}")
    print(f"   🔢 Total de usos: {learning_stats.get('total_usage', 0)}")
    print(f"   💾 Uso de memoria: {learning_stats.get('memory_usage', '0/100')}")
    
    # Mostrar patrones más usados
    top_patterns = learning_stats.get('top_patterns', [])
    if top_patterns:
        print(f"\n   🏆 Top 5 Patrones Más Usados:")
        for i, pattern in enumerate(top_patterns, 1):
            label = pattern.get('label', 'Desconocido')
            usage_count = pattern.get('usage_count', 0)
            print(f"      {i}. {label} - usado {usage_count} veces")
    
    # Guardar la memoria de aprendizaje
    print(f"\n4️⃣ Guardando memoria de aprendizaje...")
    metamodulo.fingerprint_system.learning_memory.save_memory()
    
    print("\n" + "=" * 70)
    print("🎉 DEMO COMPLETADO EXITOSAMENTE!")
    print("🚀 El Sistema de Aprendizaje está funcionando perfectamente.")
    print("💡 La IA ahora puede:")
    print("   ✅ Aprender patrones nuevos automáticamente")
    print("   🔍 Reconocer patrones repetidos")
    print("   🏷️ Generar etiquetas inteligentes")
    print("   📚 Mantener memoria persistente")
    print("   🔄 Mejorar con cada uso")

if __name__ == "__main__":
    try:
        demo_learning_system()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\n❌ Error en el demo: {e}")
        import traceback
        traceback.print_exc()
