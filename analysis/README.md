# analysis/ - Analisis de Experimentos

Notebooks de Jupyter y scripts para analizar resultados de entrenamiento, comparar modelos y estudiar hiperparametros.

## Experimentos

### BT1/ - Analisis Bradley-Terry
Ranking de checkpoints usando el modelo Bradley-Terry.
- `03_eval_BT01__tops.py`: Evalua top 30 checkpoints en torneo
- `eval_01_vs_baselines.py`: Compara contra baselines (bot_good, bot_random, bot_Michael)
- Notebooks: Comparacion de pesos, standings del torneo, analisis de resultados

### E02_win_rate/ - Sensibilidad al Learning Rate
Estudio con 4 variantes de learning rate:
- **base**: LR = 1e-4
- **A**: LR = 1e-5
- **B**: LR = 5e-5
- **C**: LR = 5e-4
- Incluye validacion por torneo suizo (R200, R5)

### E02_wr/ - Win Rate Extendido
Extension del E02 con torneos mas largos (R1k-5k).
- Comparacion de pesos entre variantes
- Torneos suizos a mayor escala

### E03_BS/ - Impacto del Batch Size
Variantes de batch size probadas: 32, 64, 512, 1024, 2048.
- Analisis completo con torneo suizo por configuracion

### E04_N_LAST_STATES/ - Ventana de Estados
Estudio del parametro `N_LAST_STATES` (cuantos estados del historial usar).
- Conclusion: Curriculum learning NO funciona con la arquitectura actual
- N=2 (sin curriculum) es lo mas estable

### E05_HYPERPARAMETER_ANALYSIS/ - Busqueda de Hiperparametros
Busqueda principiada con justificacion teorica.
- Variantes: E05a (500 games, BS=128), E05b (1000, BS=256), E05c (2000, BS=512)
- Incluye guia teorica con calculos de cobertura del espacio de estados

### E06_architecture_prebug/ - Comparacion de Arquitecturas
Evaluacion de variantes de red neuronal (pre-correccion de bugs).
- Visualizacion de win rate y dinamica de pesos

### E08_loss-balance/ - Balance de Loss
Comparacion de funciones de loss via torneos suizos.
- `combined_avg` vs `only_select` vs `only_place`

### legacy/ - Analisis Archivados
Analisis tempranos ya no activos.

### XYZ/ - Experimental
Datos experimentales auxiliares.
