# ########################### DEBUG
# When True, checks for winning in 2x2 squares. False, only in rows, columns and diagonals.
mode_2x2 = True

# every epoch experience is generated with a new bot instance, models are saved at the end of each epoch
BATCH_SIZE = 19
EPOCHS = 100

# number of times the network is updated per epoch
ITER_PER_EPOCH = 20
MATCHES_PER_EPOCH = 100
STEPS_PER_EPOCH = 100_0  # ~x10 of matches_per_epoch, used to generate experience

REPLAY_SIZE = 3_00  # ~x3 STEPS_PER_EPOCH, info from last 3 epochs

# update target network every n batches processed, ~1/3 of ITER_PER_EPOCH
N_BATCHS_2_UPDATE_TARGET = 30

N_MATCHES_EVAL = 5  # number of matches to evaluate the bot at the end of each epoch for the selected previous rival

# number of last states to consider in the experience generation at the beginning of training
N_LAST_STATES_INIT: int = 3
# number of last states to consider in the experience generation at the end of training. -1 means all states
N_LAST_STATES_FINAL: int = 16
# temperature for exploration, higher values lead to more exploration
TEMPERATURE_EXPLORE = 2

# temperature for exploitation, lower values lead to more exploitation
TEMPERATURE_EXPLOIT = 0.1

N_PLAYERS_PLOT = 4

RIVALS_IN_TOURNAMENT = 15  # number of rivals to evaluate the bot against in the contest at the end of each epoch
POINTS_BY_RIVAL = 6
