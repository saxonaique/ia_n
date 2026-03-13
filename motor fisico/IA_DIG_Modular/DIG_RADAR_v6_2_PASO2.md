# 📊 DIG RADAR v6.2 — Panel de visualización integrado

## Objetivo
Integrar el vector `Spain_EU_Economy` en la visualización del **Paso 2** para mostrar tres capas simultáneas:

1. Militar
2. Energética
3. Económica (España/UE)

---

## 1️⃣ Gráfico de presión sistémica (score global)

Evolución prevista según el LSTM:

| Tiempo | Score |
| ------ | ----- |
| Ahora  | **25.0** |
| +6 h   | 23.87 |
| +12 h  | 23.57 |
| +18 h  | 23.92 |
| +24 h  | 23.96 |
| Día 7  | **23.81** |

**Interpretación rápida**

- El sistema sigue en **zona roja extrema**.
- No aparece una desescalada natural en el horizonte de una semana.
- La caída de divergencia (0.651 → 0.583) indica **reorganización del sistema**, no relajación.

---

## 2️⃣ Panel de vectores críticos

### 🟥 Militar

| Vector | Probabilidad 24h |
| ------ | ---------------- |
| ☢️ Fordow nuclear | **99%** |
| 🇬🇧 UK involvement | **99%** |
| 🇾🇪 Yemen/Houthi | **98%** |

**Interpretación**

- El sistema espera **expansión del conflicto**, no congelación.

### 🟧 Energía global

| Vector | Probabilidad |
| ------ | ------------ |
| Petróleo >100$ | **97%** |
| Ataques en Hormuz | **94%** |
| Interrupción LNG | **92%** |

Esto explica por qué los vectores europeos se disparan.

---

## 3️⃣ 🇪🇸 Spain_EU_Economy — Visualización estratégica

Probabilidades comparadas:

| Evento | 7 días | 30 días |
| ------ | ------ | ------- |
| Inflación >3% | **95%** | 89% |
| ERTE transporte | **91%** | 77% |
| Medidas emergencia combustible | **96%** | 91% |
| Recesión técnica eurozona | **86%** | 67% |
| Crisis política bases | **83%** | 63% |
| Subida tipos BCE | **82%** | 60% |
| Crisis LNG invierno | **88%** | 72% |

---

## 🧠 Lectura estratégica del sistema

Las tres capas están acopladas:

1. **Guerra → energía**: el conflicto presiona petróleo y gas.
2. **Energía → economía europea**: inflación y transporte son los primeros en sufrir.
3. **Economía → política interna**: bases militares y decisiones del BCE entran en tensión.

En lenguaje DIG, el sistema está en un **estado de máxima energía con redistribución de tensiones**: la curvatura del campo informacional no desaparece, **solo se desplaza hacia nuevos vectores**.

---

## 🔎 Lo más importante para España ahora

El vector con mayor impacto real es:

**Crisis combustible + transporte**

Razones:

- España depende mucho del **transporte por carretera y pesca**.
- Un gasóleo >2 €/L activa protestas y ERTE muy rápido.
- Esto ya ocurrió parcialmente en **2022**, pero con un conflicto mucho menor.

---

## ✅ Estado actual del DIG Radar

```text
17 vectores activos
3 frentes militares nuevos
7 vectores España-UE
Score global: 25/25
Sistema estable y predicción operativa
```

---

## Próximas mejoras sugeridas

1. **Mapa mundial de frentes** (tipo sala de guerra).
2. **Motor de detección automática de nuevos frentes**.
3. **Simulación de escenarios** (p. ej., cierre de Hormuz).
