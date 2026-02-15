# E02_win_rate/ - Sensibilidad al Learning Rate

Experimento comparando 4 learning rates para encontrar el optimo.

## Variantes
| Variante | Learning Rate |
|----------|--------------|
| base | 1e-4 |
| A | 1e-5 |
| B | 5e-5 |
| C | 5e-4 |

## Notebooks
- `01_analysis_comparison.ipynb`: Comparacion de curvas de aprendizaje y convergencia
- `02b_compare_with_swiss_R200.ipynb`: Validacion por torneo suizo (200 rondas)
- `02b_compare_with_swiss_r5.ipynb`: Torneo suizo rapido (5 rondas)
- `v_epoch_batch.ipynb`: Analisis de epoch vs batch

## Datos
- `E02_win_rate*.pkl`: Resultados de entrenamiento por variante
- `swiss_tournament_*.pkl`: Resultados de torneos suizos
- `*.log`: Logs de entrenamiento por variante
