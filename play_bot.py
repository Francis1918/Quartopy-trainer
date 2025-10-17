# -*- coding: utf-8 -*-
"""
Python 3
03 / 10 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""
from bot.human import Quarto_bot as Human_bot
from quartopy import play_games
from bot.CNN_bot import Quarto_bot

human = Human_bot()

# better
f = "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt"
# medio malo
# f = "CHECKPOINTS//EXP_id03//20250922_1920-EXP_id03_epoch_0377.pt"

bot = Quarto_bot(model_path=f)
_, win_rate_p1 = play_games(
    matches=1,
    player1=bot,
    player2=human,
    verbose=True,
    save_match=True,
    mode_2x2=True,
)
