# E05_HYPERPARAMETER_ANALYSIS/ - Busqueda Principiada de Hiperparametros

Busqueda de hiperparametros con justificacion teorica basada en cobertura del espacio de estados.

## Variantes
| Config | Matches/Epoch | Batch Size | Estados/Epoch |
|--------|--------------|------------|---------------|
| E05a | 500 | 128 | 2,000 |
| E05b | 1,000 | 256 | 4,000 |
| E05c | 2,000 | 512 | 8,000 |

## Marco Teorico
- Quarto tiene ~10^8 estados posibles
- E05b cubre 0.004% por epoca
- Se necesitan ~125 epocas para cubrir 1% del espacio

## Archivos
- `E05_analysis.ipynb`: Analisis principal comparando configuraciones
- `E05_weight_analysis.ipynb`: Analisis de distribucion de pesos
- `E05_EXPERIMENT_PLAN.md`: Plan del experimento con teoria
- `E05_comparison_metrics.csv`: Metricas de comparacion
- `E05_statistical_tests.csv`: Tests estadisticos entre variantes
