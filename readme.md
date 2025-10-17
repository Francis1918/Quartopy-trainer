# Hierarchical-SAE: Quarto Bot Competition Framework

## Overview
This project provides a framework for training and evaluating AI bots that play the board game Quarto. It uses a hierarchical structure with Sparse Auto-Encoders (SAE) to develop intelligent game-playing agents.

## Features
- Train CNN-based Quarto bots with different architectures
- Play matches between different bot models
- Evaluate bot performance with detailed statistics
- Support for deterministic and non-deterministic play styles

## Requirements
- Python 3.x
- quartopy (Quarto game engine)
- PyTorch (for CNN models)
- Other dependencies as specified in requirements.txt

## Project Structure
- `bot/` - Contains bot implementation files
  - `CNN_bot.py` - Standard CNN bot implementation
  - `CNN_F_bot.py` - Francis' bot implementation with modifications
- `CHECKPOINTS/` - Contains trained model weights
  - `EXP_id03/` - Experiment ID 3 model checkpoints
  - `others/` - Additional model checkpoints

## Bot Models
The project includes several pre-trained bots:
1. **bot_malo** - Basic model with limited capabilities
2. **bot_rand** - Random move bot (baseline)
3. **bot_good** - Improved model with better gameplay
4. **bot_francis** - Francis' implementation with custom architecture
5. **bot_michael** - Michael's trained model implementation

## Playing Between Bots
The `play_between_bots.py` script allows you to run matches between different bot implementations:

```python
# Example usage
from quartopy import play_games
from bot.CNN_bot import Quarto_bot

# Load bots
bot_A = Quarto_bot(model_path="path/to/model.pt", deterministic=False, temperature=0.1)
bot_B = Quarto_bot(model_path="path/to/another_model.pt", deterministic=True)

# Play matches
results, win_rate = play_games(
    matches=500,
    player1=bot_A,
    player2=bot_B,
    verbose=False,
    save_match=False,
    mode_2x2=True
)

print(win_rate)
```

### Configuration Options
- `deterministic` - Whether the bot makes deterministic decisions
- `temperature` - Controls randomness in non-deterministic decision making
- `N_MATCHES` - Number of matches to play
- `VERBOSE` - Whether to print detailed match information
- `mode_2x2` - Use 2x2 game mode (standard Quarto is 4x4)

## Results Interpretation
The script outputs win rates for each bot when playing as Player 1 and Player 2, helping to identify any first-player advantage and overall bot performance.

## Future Work
- Implementation of reinforcement learning approaches
- Hybrid models combining different techniques
- Tournament system for multiple bots
