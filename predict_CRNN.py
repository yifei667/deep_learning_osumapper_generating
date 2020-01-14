import os, torch, time
import numpy as np
from model import ConvLstmNet, ConvLstmNet2
from train_utils import Trainer
from torch.autograd import Variable
from lib import Data2Torch, load_data
from osureader import read_and_save_osu_tester_file
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_npz_p(fn):
    with np.load(fn) as data:
        wav_data = data["wav"];
        wav_data = np.swapaxes(wav_data, 2, 3);
        ticks = data["ticks"];
        timestamps = data["timestamps"];
        extra = data["extra"];

        # Extra vars
        bpms = extra[0];
        slider_lengths = extra[1];
        ex1 = (60000 / bpms) / 500 - 1;
        ex2 = bpms / 120 - 1;
        ex3 = slider_lengths / 150 - 1;

        div_data = np.array([divisor_array_p(k) + [ex1[k], ex2[k], ex3[k]] for k in ticks]);
    return wav_data, div_data, ticks, timestamps;


def divisor_array_p(k):
    d_range = list(range(0, divisor));
    return [int(k % divisor == d) for d in d_range];

def read_new_map(file_path, fn):
    start = time.time()
    read_and_save_osu_tester_file(file_path.strip(), filename=fn, divisor=divisor)
    end = time.time()
    print("Map data saved! time = " + str(end - start) + " secs.")

def read_one_osu_test(osuFile):
    fn = "mapthis"
    read_new_map(osuFile, fn)
    fn = fn + ".npz"
    test_data, div_data, ticks, timestamps = read_npz_p(fn)
    test_data = test_data[:,:,:,0]

    # Make time intervals from test data
    if test_data.shape[0] % time_interval > 0:
        test_data = test_data[:-(test_data.shape[0] % time_interval)]
        div_data = div_data[:-(div_data.shape[0] % time_interval)]
    test_data2 = np.reshape(test_data, (-1, time_interval, test_data.shape[1], test_data.shape[2]))
    div_data2 = np.reshape(div_data, (-1, time_interval, div_data.shape[1]))

    return (test_data2, div_data2)


def predict_CRNNs(osuFile, model_1, model_2):

    # read .osu
    test_data = read_one_osu_test(osuFile)

    # load models
    model1 = ConvLstmNet()
    model2 = ConvLstmNet2()
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
    model1.load_state_dict(torch.load(model_1)['state_dict'])
    model2.load_state_dict(torch.load(model_2)['state_dict'])

    # prepare dataloader (function inside lib.py)
    t_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}

    te_loader = torch.utils.data.DataLoader(Data2Torch(test_data, lim), **t_kwargs)

    print(len(te_loader))

    model1.eval()
    model2.eval()
    pred1 = []
    pred2 = []

    for batch_idx, _input in enumerate(te_loader):

        spec, div, _ = Variable(_input[0]).to(device), Variable(_input[1]).to(device), Variable(_input[2]).to(
            device)

        outputs1 = model1(spec, div)
        outputs2 = model2(spec, div)

        pred1.extend(outputs1.data.cpu().numpy())
        pred2.extend(outputs2.data.cpu().numpy())

    #print(pred2)
    return np.array(pred1), np.array(pred2), test_data[1]

if __name__ == "__main__":
    osuFile = "maps/317749 EYE_XY feat. Yoneko - Knight of Firmament/EYE_XY feat. Yoneko - Knight of Firmament (Pho) [Normal].osu"
    model_1 = 'models/20191128/ConvLstm_batch1_lr0.01/model'
    model_2 = 'models/20191130/ConvLstm2_batch1_lr0.001/model'
    predict_CRNNs(osuFile, model_1, model_2)