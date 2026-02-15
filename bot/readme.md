# bot/ - Implementaciones de Jugadores

Contiene las distintas implementaciones de jugadores (bots) que interactuan con el motor de juego Quartopy.

## Archivos

### CNN_bot.py
Bot principal basado en redes neuronales CNN. Envuelve cualquier modelo que herede de `NN_abstract`.

- **Clase**: `Quarto_bot(BotAI)`
- **Parametros clave**:
  - `model_path`: Ruta a pesos pre-entrenados (.pt)
  - `model`: Instancia de modelo en memoria
  - `model_class`: Clase del modelo (default: `QuartoCNN`)
  - `deterministic`: `True` = argmax, `False` = muestreo por temperatura
  - `temperature`: Control de aleatoriedad (0.1 = greedy, 2.0 = exploratorio)
- **Metodos principales**:
  - `place_piece(game, piece)` -> coordenadas del tablero
  - `select(game)` -> pieza para el oponente
  - `evaluate(exp_batch)` -> Q-values para entrenamiento

### CNN_F_bot.py
Variante del CNN bot con manejo robusto de errores y validacion de dependencias. Usado con la arquitectura `QuartoCNNExtended`.

- Validacion exhaustiva de imports de Quartopy
- Deteccion automatica de dispositivo (GPU/CPU)
- Mensajes de error detallados para debugging

### random_bot.py
Bot baseline que juega movimientos aleatorios validos.

- `select(game)` -> pieza aleatoria del storage
- `place_piece(game, piece)` -> posicion aleatoria valida

### human.py
Interfaz interactiva para jugar como humano contra bots.

- Muestra piezas/posiciones disponibles con indices
- Acepta input por consola
- Validacion de entrada con valores por defecto

## Uso

```python
from bot.CNN_bot import Quarto_bot

# Cargar bot entrenado
bot = Quarto_bot(model_path="CHECKPOINTS/05_LOSS/mi_modelo.pt", temperature=0.1)

# Bot con modelo en memoria
bot = Quarto_bot(model=mi_modelo, deterministic=False, temperature=2.0)
```
