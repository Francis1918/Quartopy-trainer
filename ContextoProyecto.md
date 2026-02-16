# Contexto del Proyecto: Quarto Bot Training Framework

**IMPORTANTE: Solo se deben modificar los parametros de entrenamiento en `trainRL.py` y `trainRL_resume_latest.py`. NO modificar la arquitectura de la red, los bots, ni las funciones de RL.** Los unicos cambios permitidos son los hiperparametros dentro de la seccion de configuracion de estos dos scripts.

## 1. El Juego: Quarto

Quarto es un juego de mesa para 2 jugadores sobre un tablero de **4x4** con **16 piezas unicas**. Cada pieza tiene 4 atributos binarios (alto/bajo, claro/oscuro, redondo/cuadrado, hueco/solido), dando 2^4 = 16 combinaciones.

**Mecanica de turnos** (lo que hace unico a Quarto):
1. Tu oponente **selecciona** una pieza y te la da
2. Tu **colocas** esa pieza en una celda vacia del tablero
3. Luego tu **seleccionas** una pieza para darle al oponente

**Condiciones de victoria:**
- Formar una linea de 4 piezas (fila, columna o diagonal) que compartan al menos 1 atributo
- Con `mode_2x2=True` (habilitado en este proyecto): tambien gana formando un cuadrado 2x2 donde las 4 piezas comparten al menos 1 atributo

---

## 2. Arquitectura de la Red: QuartoCNN_uncoupled

La CNN tiene **~77,168 parametros** y dos cabezas independientes:

### Entrada
- `x_board`: `(batch, 16, 4, 4)` — codificacion one-hot del tablero (16 canales, uno por pieza posible)
- `x_piece`: `(batch, 16)` — vector one-hot de la pieza que el oponente te dio

### Backbone compartido
```
x_piece → fc_in_piece(16→16) → reshape a (1,4,4) → concat con x_board → (17,4,4)
→ Conv2d(17→16, 3x3, padding=1) + ReLU
→ Conv2d(16→32, 3x3, padding=1) + ReLU
→ Flatten(32*4*4=512)
→ Linear(512→128) + ReLU
```

### Dos cabezas independientes (por eso "uncoupled")
- **Place head**: `Linear(128→16) + tanh` → Q-values para cada celda del tablero (donde colocar)
- **Select head**: `Linear(128→16) + tanh` → Q-values para cada pieza (cual darle al oponente)

Ambas salidas estan en el rango **[-1, +1]** gracias a `tanh`.

---

## 3. Proceso de Entrenamiento (DQN)

### 3.1 Generacion de Experiencia (Self-Play)

Cada epoca:
1. La red juega **100 partidas contra si misma** (`MATCHES_PER_EPOCH=100`)
2. De cada partida se guardan los **ultimos 2 estados** (`N_LAST_STATES=2`), enfocando el aprendizaje en posiciones de final de juego
3. Esto genera `2 * 100 = 200` transiciones por epoca (`STEPS_PER_EPOCH`)
4. Durante self-play se usa `TEMPERATURE_EXPLORE=2` (alta exploracion via softmax suave)

Cada transicion contiene: `(estado_tablero, pieza_dada, accion_colocar, accion_seleccionar, recompensa, siguiente_estado)`

### 3.2 Funcion de Recompensa: "propagate"

Con `REWARD_FUNCTION="propagate"`, **todos los estados** de una partida reciben la recompensa final:
- Estados del ganador: `R = +1`
- Estados del perdedor: `R = -1`
- Empate: `R = 0`

(Alternativas no usadas: `"final"` solo recompensa los 2 ultimos estados; `"discount"` aplica decaimiento exponencial `0.8^t` hacia atras)

### 3.3 Paso DQN (Training Step)

Se realizan **6 iteraciones** por epoca (`ITER_PER_EPOCH = 200 // 30`):

1. Muestrear batch de 30 transiciones del replay buffer (capacidad: 20,000)
2. Calcular Q-values del modelo actual para las acciones tomadas
3. Con `LOSS_APPROACH="combined_avg"`:
   ```
   Q_actual = (Q_place + Q_select) / n_valid
   ```
   donde `n_valid` es 1 o 2 dependiendo de si ambas acciones son validas (primer movimiento no tiene placement, ultimo movimiento no tiene selection)
4. Calcular target con red objetivo (Bellman):
   ```
   Q_target = reward + 0.99 * max(Q_target_net(s'))
   ```
5. Loss = `SmoothL1Loss(Q_actual, Q_target)`
6. Gradient clipping a `MAX_GRAD_NORM=1.0`
7. **Target network** se actualiza via soft update (`TAU=0.01`) cada 2 batches

### 3.4 Evaluacion

Cada epoca, el modelo juega **30 partidas** contra cada baseline:
- **bot_loss-BT**: Un modelo entrenado previamente (epoch 1034 del experimento `LOSS_APPROACHs_1212-2_only_select`)
- **bot_random**: Un modelo sin entrenar (pesos aleatorios)

Se usa `TEMPERATURE_EXPLOIT=0.1` (decision casi determinista) para evaluar.

---

## 4. Interpretacion de las Graficas

### 4.1 Win Rate

- **Eje X**: Epoca de entrenamiento
- **Eje Y**: Tasa de victoria (0.0 a 1.0)
- **Puntos dispersos**: Win rate crudo de cada epoca (30 partidas)
- **Linea suave**: Media movil con ventana de 10 epocas
- **Banda sombreada**: +/-1 desviacion estandar de la ventana
- **Dos series**: vs `bot_loss-BT` (naranja) y vs `bot_random` (azul)
- **Que buscar**: Win rate subiendo indica aprendizaje. Contra random deberia subir primero. Un win rate de ~0.5 contra random indica que el modelo no es mejor que el azar.

### 4.2 Loss (Training Loss)

- **Eje X**: Epoca
- **Eje Y**: Valor del loss (SmoothL1)
- **Linea azul**: Media del loss por epoca (promedio de las 6 iteraciones DQN)
- **Banda celeste**: +/-1 desviacion estandar
- **Que buscar**: Bajada inicial rapida seguida de estabilizacion. Si el loss sube gradualmente, puede indicar inestabilidad. La banda ancha indica alta varianza entre batches.

### 4.3 Q-values (Histograma 2D / Spectrograma)

Es una cuadricula de **2x3 subplots**:

|  | R = -1 (derrotas) | R = 0 (empates) | R = +1 (victorias) |
|--|---|---|---|
| **Q_place** | Superior izquierda | Superior centro | Superior derecha |
| **Q_select** | Inferior izquierda | Inferior centro | Inferior derecha |

Cada subplot es un **heatmap temporal**:
- **Eje X**: Epoca de entrenamiento
- **Eje Y**: Valor del Q-value (-1.1 a +1.1)
- **Color (viridis)**: Porcentaje de muestras en ese bin de Q-value
- **Que buscar**: Idealmente, los Q-values para R=+1 deberian concentrarse cerca de +1 (el modelo aprende que ganar es bueno), y para R=-1 cerca de -1 (perder es malo). Si todo se colapsa a un solo valor, el modelo no esta diferenciando.

### 4.4 Boards (Visualizacion de Estados)

- Muestra pares de tableros (estado actual vs siguiente estado) de un batch aleatorio
- Cada subplot muestra el tablero 4x4 como una grilla
- El titulo indica: `{pieza}|{accion}|{jugador} | R={recompensa}`
- **Que buscar**: Verificar que los estados son razonables y que las recompensas estan asignadas correctamente. Tableros vacios al final del entrenamiento podrian indicar un bug.

---

## 5. Decision del Bot (Temperatura)

Cuando el bot juega, convierte los Q-values en probabilidades via softmax:

```
P(accion_i) = exp(Q_i / T) / sum(exp(Q_j / T))
```

- **T = 0.1** (explotacion): Softmax muy puntiagudo, casi siempre elige la mejor accion
- **T = 2.0** (exploracion): Distribucion mas uniforme, explora mas acciones
- **Determinista**: Ignora temperatura, ordena por Q-value y elige el maximo

Si la posicion o pieza elegida no es valida (celda ocupada o pieza ya usada), se intenta con la siguiente opcion del ranking.


