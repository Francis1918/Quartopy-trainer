# Quarto Bot Training Framework

> **IMPORTANTE: Solo se deben modificar los parametros de entrenamiento en `trainRL.py` y `trainRL_resume_latest.py`. NO modificar la arquitectura de la red, los bots, ni las funciones de RL.** Los unicos cambios permitidos son los hiperparametros dentro de la seccion de configuracion de estos dos scripts.

## Overview
This project provides a comprehensive framework for training and evaluating AI bots that play the board game Quarto using Deep Q-Learning (DQN). The system supports multiple neural network architectures, loss calculation approaches, and reward functions to develop intelligent game-playing agents through reinforcement learning.

## Features
- **Deep Q-Learning Training**: Train CNN-based Quarto bots with configurable hyperparameters
- **Multiple Architectures**: Support for standard CNN and uncoupled CNN architectures
- **Flexible Loss Approaches**: 
  - `combined_avg` - Average loss from both select and place actions
  - `only_select` - Focus training on piece selection
  - `only_place` - Focus training on piece placement
- **Reward Function Options**:
  - `final` - Reward only at game end
  - `propagate` - Propagate rewards through game states
  - `discount` - Discounted reward propagation
- **Swiss Tournament System**: Evaluate multiple checkpoints with McMahon scoring
- **Automated Training**: Generate multiple training configurations for hyperparameter sweeps
- **Baseline Evaluation**: Compare trained bots against reference models
- **Comprehensive Logging**: Detailed training logs with timestamps

## Requirements
- Python 3.x
- [quartopy](https://github.com/ztjona/Quartopy) (Quarto game engine) - v1.3.2 or later
- PyTorch (for CNN models)
- TorchRL (for replay buffers and RL utilities)
- tqdm (progress bars)
- colorama (colored terminal output)
- docopt (command-line argument parsing)
- Other dependencies as specified in requirements.txt

## Project Structure
- `bot/` - Bot implementation files
  - `CNN_bot.py` - Standard CNN bot implementation
  - `CNN_F_bot.py` - Extended architecture bot
  - `human.py` - Human player interface
  - `random_bot.py` - Random baseline bot
- `models/` - Neural network architectures
  - `CNN1.py` - Standard CNN model
  - `CNN_uncoupled.py` - Uncoupled CNN (separate select/place networks)
  - `CNN_fdec.py` - Extended CNN architecture
- `CHECKPOINTS/` - Trained model weights organized by experiment
  - `05_LOSS/` - Latest loss approach experiments
  - `BT_0/`, `BT_1/` - Bradley-Terry experiments
  - `E02_win_rate/` - Win rate optimization experiments
  - `EXP_id03/` - Early experimental checkpoints
- `QuartoRL/` - Reinforcement learning utilities
  - `RL_functions.py` - Core RL training functions
  - `contest.py` - Bot competition functions
  - `plotting.py` - Visualization utilities
- `analysis/` - Experiment analysis notebooks and scripts
- `logs/` - Training log files with timestamps
- `train_scripts/` - Auto-generated training configurations

## Training System

### Main Training Script (`trainRL.py`)
Configure and run DQN training with the following key parameters:

```python
# Architecture
ARCHITECTURE = QuartoCNN_uncoupled  # or QuartoCNN

# Loss and Reward Configuration
LOSS_APPROACH = "combined_avg"  # "combined_avg", "only_select", "only_place"
REWARD_FUNCTION = "propagate"   # "final", "propagate", "discount"

# Training Parameters
EPOCHS = 3000
BATCH_SIZE = 30
LR = 5e-5
GAMMA = 0.99
TAU = 0.01

# Experience Generation
MATCHES_PER_EPOCH = 100
N_LAST_STATES_FINAL = 2  # Number of recent states to learn from

# Exploration/Exploitation
TEMPERATURE_EXPLORE = 2     # Higher = more exploration
TEMPERATURE_EXPLOIT = 0.1   # Lower = more exploitation
```

### Running Training
```bash
# Direct execution
./runpy.sh trainRL.py

# Training runs are logged to logs/ with timestamps
# Models are saved to CHECKPOINTS/<EXPERIMENT_NAME>/
```

### Automated Training Configuration (`run_trains.py`)
Generate multiple training scripts with different parameters:

```bash
python run_trains.py
# Creates train_*.py files in train_scripts/
# Then execute them with:
./runpy.sh train_scripts/train_LOSS_APPROACH_1213-1_combined_avg.py &
./runpy.sh train_scripts/train_LOSS_APPROACH_1213-2_only_select.py &
./runpy.sh train_scripts/train_LOSS_APPROACH_1213-3_only_place.py
```

## Bot Models
The project includes various trained bot checkpoints:
1. **bot_loss / bot_loss_BT** - Models trained with loss-focused approaches (Bradley-Terry)
2. **bot_rand** - Random move bot (baseline for comparison)
3. **bot_good / bot_good2B** - Well-performing models from win rate experiments
4. **bot_Francis_dec** - Extended architecture implementation
5. **bot_Michael** - Alternative training approach models

## Evaluating Bots

### Head-to-Head Matches (`play_between_bots.py`)
Compare two bot models directly:

```python
from quartopy import play_games
from bot.CNN_bot import Quarto_bot
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled

# Load bots with specific architectures
bot_A = Quarto_bot(
    model_path="CHECKPOINTS/05_LOSS/model.pt",
    model_class=QuartoCNN_uncoupled,
    deterministic=False,
    temperature=0.1
)

bot_B = Quarto_bot(
    model_path="CHECKPOINTS/BT_1/baseline.pt",
    model_class=QuartoCNN_uncoupled,
    deterministic=False,
    temperature=0.1
)

# Play matches with color swap for fairness
res1, win_rate_p1 = play_games(matches=500, player1=bot_A, player2=bot_B, mode_2x2=True)
res2, win_rate_p2 = play_games(matches=500, player1=bot_B, player2=bot_A, mode_2x2=True)
```

### Configuration Options
- `model_class` - Specify architecture (QuartoCNN, QuartoCNN_uncoupled, QuartoCNNExtended)
- `deterministic` - Whether the bot makes deterministic decisions
- `temperature` - Controls randomness in non-deterministic decision making (0.1 = exploitative, 2.0 = exploratory)
- `mode_2x2` - Enable 2x2 square win condition checking (in addition to standard lines)

### Swiss Tournament System (`run_swiss_tournament.py`)
Evaluate all checkpoints in a folder using Swiss-system pairing:

```bash
# Basic usage
python run_swiss_tournament.py CHECKPOINTS/E02_win_rate/ --rounds=200

# Single Swiss (no color swap)
python run_swiss_tournament.py CHECKPOINTS/05_LOSS/ --rounds=100 --single

# Without McMahon scoring (no initial points based on epoch)
python run_swiss_tournament.py CHECKPOINTS/BT_1/ --rounds=50 --no-mcmahon

# Custom temperature and output file
python run_swiss_tournament.py CHECKPOINTS/05_LOSS/ --temperature=0.5 --results-file=my_tournament.pkl
```

#### Swiss Tournament Features
- **Swiss Pairing**: Players with similar scores face each other
- **McMahon Scoring**: Initial points based on model epoch (estimated strength)
- **Double Swiss**: Each pairing plays twice with color swap (default)
- **Results Export**: Tournament data saved as pickle file with timestamps

### Playing Against Human (`play_bot.py`)
Test your bot against human players:

```python
from bot.human import Quarto_bot as Human_bot
from bot.CNN_bot import Quarto_bot

human = Human_bot()
bot = Quarto_bot(model_path="CHECKPOINTS/05_LOSS/best_model.pt")

play_games(matches=1, player1=bot, player2=human, verbose=True, save_match=True, mode_2x2=True)
```

## Results Interpretation
- **Win Rate**: Percentage of games won (check both player1 and player2 positions)
- **First-Player Advantage**: Compare win rates when playing as different colors
- **Swiss Tournament Rankings**: Final standings show relative strength across all checkpoints
- **Loss Plots**: Training progress visualization (saved to CHECKPOINTS folder)
- **Q-Value Tracking**: Monitor Q-value distributions during training

## Logging
All training runs create timestamped logs in the `logs/` directory:
```
logs/trainRL-24-12-14_10_30.log
```

The `runpy.sh` script automatically:
- Creates log files with timestamps
- Displays output in terminal (with colors)
- Strips ANSI color codes from saved logs for readability

## Reglas de Modificacion

### PERMITIDO - Solo parametros en `trainRL.py` y `trainRL_resume_latest.py`:
```python
STARTING_NET          # Ruta al checkpoint inicial (None = pesos aleatorios)
EXPERIMENT_NAME       # Nombre del experimento
ARCHITECTURE          # QuartoCNN o QuartoCNN_uncoupled (seleccion, NO modificar las clases)
LOSS_APPROACH         # "combined_avg", "only_select", "only_place"
REWARD_FUNCTION       # "final", "propagate", "discount"
GEN_EXPERIENCE_BY_EPOCH  # True/False
N_MATCHES_EVAL        # Partidas de evaluacion por epoca
BATCH_SIZE            # Tamano del batch
EPOCHS                # Numero de epocas
N_LAST_STATES_INIT    # Estados iniciales del historial
N_LAST_STATES_FINAL   # Estados finales del historial
MATCHES_PER_EPOCH     # Partidas de self-play por epoca
TEMPERATURE_EXPLORE   # Temperatura de exploracion
TEMPERATURE_EXPLOIT   # Temperatura de explotacion
FREQ_EPOCH_SAVING     # Frecuencia de guardado
MAX_GRAD_NORM         # Clipping de gradientes
LR / LR_F             # Learning rate inicial y final
TAU                   # Tasa de soft update del target network
GAMMA                 # Factor de descuento
BASELINES             # Lista de rivales para evaluacion
```

### PROHIBIDO - No modificar:
- `models/` - Arquitecturas de redes neuronales
- `bot/` - Implementaciones de bots
- `QuartoRL/` - Funciones de RL (gen_experience, DQN_training_step, etc.)
- `utils/` - Utilidades del proyecto
- Logica del loop de entrenamiento dentro de trainRL.py

## Advanced Features
- **Replay Buffer**: Experience replay with configurable size
- **Target Network**: Soft updates with TAU parameter
- **Gradient Clipping**: Prevents exploding gradients (MAX_GRAD_NORM)
- **Learning Rate Scheduling**: Cosine annealing from LR to LR_F
- **Baseline Evaluation**: Automatic evaluation against reference bots every epoch
- **Checkpoint Management**: Periodic model saving with configurable frequency
