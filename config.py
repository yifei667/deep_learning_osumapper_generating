map_path = "maps/"
mapdata_path = "mapdata/"
level = 'Hard'
data_path = "mapdata/{}/".format(level)

divisor=4
time_interval = 16
time_interval2 = 64

lim = -1#10000

# training parameters
batch_size = 1
num_workers = 1  # fixed
shuffle = True  # fixed
epoch = 1000  # fixed
lr = 0.001
model_choose = 'ConvLstm2'

# momentum max & min
train_glob_max = [3.09358387, 0.06165558]
train_glob_min = [ 0., -0.05206194]

# Hard
train_glob_mean = [3.53243368e-01, -2.09375834e-05]
train_glob_std = [0.19857471, 0.01427526]

# predict
dist_multiplier = 1
note_density = {"Easy": 0.3, "Normal": 0.3, "Hard": 0.4}
slider_favor = 0
divisor_favor = [0] * divisor

# model paths
model_CRNN1 = {"Easy": "models/2019121/ConvLstm_Easy_batch1_lr0.001/model", \
               "Normal": "models/20191128/ConvLstm_batch1_lr0.01/model", \
               "Hard": "models/2019121/ConvLstm_Hard_batch1_lr0.001/model"}
model_CRNN2 = {"Easy": "models/2019121/ConvLstm2_Easy_batch1_lr0.001/model", \
               "Normal": "models/20191130/ConvLstm2_batch1_lr0.001/model", \
               "Hard": "models/2019121/ConvLstm2_Hard_batch1_lr0.001/model"}