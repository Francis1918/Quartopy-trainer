# QuartoRL/ - Funciones de Reinforcement Learning

Modulo principal con las funciones de entrenamiento DQN, generacion de experiencia, evaluacion y visualizacion.

## Archivos

### RL_functions.py - Core del Entrenamiento

**`gen_experience(p1_bot, p2_bot, ...)`**
Genera experiencia haciendo que dos bots jueguen entre si (self-play).
- `n_last_states`: Cuantos estados recientes guardar por partida (default: 16)
- `number_of_matches`: Total de partidas a jugar
- `REWARD_FUNCTION_TYPE`: "propagate" | "final" | "discount"
- Retorna: `TensorDict` con `state_board`, `state_piece`, `action_place`, `action_sel`, `reward`, `done`, `next_state_*`

**`DQN_training_step(policy_net, target_net, GAMMA, exp_batch, ...)`**
Un paso de entrenamiento DQN sobre un batch de experiencias.
- `LOSS_APPROACH`: "combined_avg" (recomendado) | "only_select" | "only_place"
- Retorna: `(state_action_values, expected_state_action_values)` para calcular loss

**`convert_2_state_action_reward(match_data, ...)`**
Convierte datos de partida de Quartopy a formato estado-accion-recompensa.

### contest.py - Evaluacion por Torneo

**`run_contest(player, rivals, ...)`**
Torneo del bot contra multiples rivales. Juega como P1 y P2 (mitad y mitad).

**`contest_2_win_rate(contest_results)`**
Convierte conteos a win rate: `(wins + 0.5*draws) / total`

### plotting.py - Visualizacion

- `plot_win_rate()` - Win rate por epoca con suavizado y bandas de error
- `plot_loss()` - Loss promedio por epoca con std
- `plot_contest_results()` - [Legacy] Heatmap de win rate

### observers.py - Observadores de Q-values

- `plot_boards_comp()` - Pares de estados del tablero lado a lado
- `plot_Qv_progress()` - Evolucion de Q-values agrupados por resultado (R=-1, 0, +1)

## Flujo de Uso

```
trainRL.py
    |-> gen_experience()        # Self-play -> TensorDict
    |-> DQN_training_step()     # Bellman update
    |-> run_contest()           # Evaluacion vs baselines
    |-> plot_win_rate()         # Graficar progreso
    |-> plot_loss()             # Graficar loss
    |-> plot_Qv_progress()      # Analizar Q-values
```
