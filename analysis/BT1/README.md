# BT1/ - Analisis Bradley-Terry (Ronda 1)

Ranking de checkpoints del entrenamiento usando el modelo Bradley-Terry y evaluacion contra baselines.

## Scripts
- `03_eval_BT01__tops.py`: Evalua los top 30 checkpoints en formato torneo
- `eval_01_vs_baselines.py`: Compara checkpoints BT1 vs baselines (bot_good, bot_random, bot_Michael)

## Notebooks
- `_bt_test.ipynb`: Pruebas del modelo Bradley-Terry
- `_weight_comp.ipynb`: Comparacion de pesos de modelos
- `02_comp_weights.ipynb`: Analisis detallado de pesos
- `02B_BT_BT1.ipynb`: Bradley-Terry aplicado a BT1
- `04_view_tops.ipynb`: Visualizacion de los mejores modelos
- `results_BT1-baselines.ipynb`: Resultados contra baselines
- `test_bradley_terry_full_bt1.ipynb`: Test completo del BT en BT1

## Datos
- `BT_1.pkl`, `p_i.pkl`, `score_norm.pkl`: Scores y resultados serializados
- `eval_BT01__tops_results.pkl`: Resultados de evaluacion top checkpoints
- `BT1_tourn.log`: Log del torneo
