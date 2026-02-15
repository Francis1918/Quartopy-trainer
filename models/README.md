# models/ - Arquitecturas de Redes Neuronales

Define las arquitecturas CNN que predicen Q-values para las acciones del juego Quarto.

## Clase Base

### NN_abstract.py
Clase abstracta que define la interfaz comun para todos los modelos.

- **Hereda de**: `ABC` + `torch.nn.Module`
- **Input**: `x_board` (batch, 16, 4, 4) + `x_piece` (batch, 16)
- **Output**: `qav_board` (batch, 16) + `qav_piece` (batch, 16)
- **Metodos**:
  - `forward()` - abstracto, cada modelo lo implementa
  - `predict()` - inferencia con modo deterministico o estocastico
  - `from_file(path)` - carga modelo desde archivo .pt
  - `export_model(suffix, folder)` - guarda checkpoint con timestamp

## Arquitecturas

### CNN1.py - Arquitectura Acoplada (Coupled)
```
Input -> fc_in_piece(16->16) -> reshape(1,4,4) + board -> concat(17 canales)
      -> conv1(17->16, k=3) -> ReLU
      -> conv2(16->32, k=3) -> ReLU
      -> flatten -> fc1(512->128) -> ReLU -> Dropout(0.5)
      |
      +-> fc2_board(128->16) -> tanh -> qav_board
      +-> fc2_piece(144->16) -> tanh -> qav_piece  [usa qav_board como input]
```
La prediccion de pieza DEPENDE de la prediccion del tablero (acoplado).

### CNN_uncoupled.py - Arquitectura Desacoplada (Recomendada)
```
Input -> fc_in_piece(16->16) -> reshape(1,4,4) + board -> concat(17 canales)
      -> conv1(17->16, k=3) -> ReLU
      -> conv2(16->32, k=3) -> ReLU
      -> flatten -> fc1(512->128) -> ReLU -> Dropout(0.5)
      |
      +-> fc2_board(128->16) -> tanh -> qav_board
      +-> fc2_piece(128->16) -> tanh -> qav_piece  [independiente]
```
Ambas cabezas son independientes. **Arquitectura preferida actual**.

### CNNfrancis.py - Variante Paralela
Misma topologia que CNN_uncoupled con override del metodo `predict()`.

### CNN_fdec.py - Arquitectura Extendida (Extended)
```
Input -> fc_in_piece -> reshape + board -> concat
      -> conv1 -> [BatchNorm] -> ReLU
      -> conv2 -> [BatchNorm] -> ReLU
      -> flatten -> fc1 -> [BatchNorm] -> ReLU -> Dropout(0.2)
      -> [fc1b -> fc1c -> fc1d] (capas opcionales)
      |
      +-> fc2_board -> logits_board
      +-> fc2_piece -> logits_piece
```
Carga dinamica de capas desde checkpoint. Soporta BatchNorm y capas opcionales.

## Comparacion

| Modelo | Acoplado | BatchNorm | Capas Dinamicas | Dropout |
|--------|----------|-----------|-----------------|---------|
| CNN1 | Si | No | No | 0.5 |
| CNN_uncoupled | No | No | No | 0.5 |
| CNNfrancis | No | No | No | 0.5 |
| CNN_fdec | No | Si | Si | 0.2 |
