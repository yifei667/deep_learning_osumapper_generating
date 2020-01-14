import os, torch
from sklearn import metrics
import numpy as np
from torch.autograd import Variable
from model import ConvLstmNet, ConvLstmNet2
from train_utils import Trainer
from lib import Data2Torch, load_data
from config import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_classification(targets, predictions):
    f1s = []
    print(predictions[0, :])
    predictions[predictions>0] = 1
    predictions[predictions<=0] = 0
    targets[targets<0] = 0
    print(predictions[0, :])
    for i in np.arange(targets.shape[1]):
        print(targets[:,i].max(),targets[:,i].min(),predictions[:,i].max(),predictions[:,i].min())

        f1 = metrics.f1_score(targets[:,i], predictions[:, i])
        f1s.append(np.round(f1,decimals=3))

    return f1s

def evaluate_regression(targets, predictions):
    r2s = []
    for i in np.arange(targets.shape[1]):
        print(targets[:,i].max(),targets[:,i].min(),predictions[:,i].max(),predictions[:,i].min())

        r2 = metrics.r2_score(targets[:,i], predictions[:, i])
        r2s.append(np.round(r2,decimals=3))

    return r2s

def evaluate_model(model, dataloader, m = 'F1'):
    model.eval()

    if m == 'R2':
        dim = 2
        interval = time_interval2
    else:
        dim = 6
        interval = time_interval

    all_predictions = np.empty((0,dim))
    all_targets = np.empty((0,dim))
    for i, (_input) in enumerate(dataloader):
        spec, div, labels = Variable(_input[0]).to(device), Variable(_input[1]).to(device), Variable(_input[2]).to(
            device)

        outputs = model(spec, div)

        outputs = outputs.view(interval, -1)
        labels = labels.view(interval, -1)

        #print(outputs.shape, labels.shape)

        all_predictions = np.concatenate((all_predictions, outputs.detach().numpy()), axis=0)
        all_targets = np.concatenate((all_targets, labels.detach().numpy()), axis=0)
        #print(all_predictions, all_targets)


    if m == 'R2':
        return evaluate_regression(np.array(all_targets), np.array(all_predictions))
    else:
        return evaluate_classification(np.array(all_targets), np.array(all_predictions))


def main():
    # model saving path
    from datetime import date
    date = date.today()
    out_model1_fn = model_CRNN1[level]
    out_model2_fn = model_CRNN2[level]

    model1 = ConvLstmNet()
    model2 = ConvLstmNet2()
    model1.to(device)
    model2.to(device)
    model1.load_state_dict(torch.load(out_model1_fn)['state_dict'])
    model2.load_state_dict(torch.load(out_model2_fn)['state_dict'])

    # model2
    if model_choose == 'ConvLstm2':
        _, test_data = load_data()

        v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
        va_loader = torch.utils.data.DataLoader(Data2Torch(test_data, lim), **v_kwargs)

        val_metrics = evaluate_model(model2, va_loader, 'R2')
        print('momentum model R-squares: ', val_metrics)
    else:
        # model1
        _, test_data = load_data()

        v_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
        va_loader = torch.utils.data.DataLoader(Data2Torch(test_data, lim), **v_kwargs)

        val_metrics = evaluate_model(model1, va_loader, 'F1')
        print('classification model F-scores: ', val_metrics)


if __name__ == "__main__":
    main()