# E04_N_LAST_STATES/ - Ventana de Estados Historicos

Estudio del parametro `N_LAST_STATES`: cuantos estados recientes del historial usar para entrenamiento.

## Hallazgo Principal
El curriculum learning (incrementar N gradualmente) NO funciona con la arquitectura CNN actual:
- N=2: Loss estable (0.026)
- N=4: Loss aumenta 14x (0.314)
- N>=8: Loss diverge severamente

## Causa Raiz
La red esta disenada para inputs de longitud fija y no maneja bien el cambio de distribucion al incrementar N.

## Recomendacion
Usar `N_LAST_STATES = 2` constante (sin curriculum).

## Archivos
- `01_E04_comprehensive_analysis.ipynb`: Analisis completo
- `E04_ANALYSIS_SUMMARY.md`: Resumen de hallazgos
