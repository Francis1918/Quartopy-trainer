# CHECKPOINTS/ - Pesos de Modelos Entrenados

Almacena los archivos `.pt` con los state_dict de los modelos entrenados, organizados por experimento.

## Convencion de nombres
```
YYYYMMDD_HHMM-<nombre_experimento>_E_<epoch>.pt
```
Ejemplo: `20260214_2254-05_LOSS_E_0000.pt` = experimento 05_LOSS, epoch 0, guardado el 14/Feb/2026.

## Subcarpetas

### 05_LOSS/
Experimentos mas recientes (Feb 2026) con diferentes configuraciones de loss.
- 5 checkpoints en epoch 0 con distintas configuraciones iniciales.

### EXP_id03/
Experimentos tempranos (Sept 2025). Baseline historico.
- Epochs: 0, 9, 377

### LOSS_APPROACHs_1212-2_only_select/
Estudio de loss approach "only_select" (Dic 2025).
- 1 checkpoint en epoch 1034

### Francis/
Prueba con arquitectura extendida `QuartoCNNExtended` (Dic 2025).
- 1 checkpoint en epoch 505

### others/
Checkpoints miscelaneos de diferentes lineas experimentales.
- Variantes de EXP_id03 y Francis

### REF/
Modelo de referencia (baseline "bueno conocido").
- Mejor modelo de E02_win_rate en epoch 22

## Uso
```python
from models.CNN_uncoupled import QuartoCNN
model = QuartoCNN.from_file("CHECKPOINTS/REF/20251023_1649-_E02_win_rate_epoch_0022.pt")
```
