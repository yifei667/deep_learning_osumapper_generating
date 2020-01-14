import torch.optim as optim
import time, sys, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboard_logger import configure, log_value

class Trainer:
    def __init__(self, model, lr, epoch, save_fn):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.criterion = nn.MSELoss()

        print('Start Training #Epoch:%d' % (epoch))

        import datetime

        current = datetime.datetime.now()

        # configure tensor-board logger
        configure('runs/' + save_fn.split('/')[-2] + '_' + current.strftime("%m:%d:%H:%M"), flush_secs=2)

    def fit(self, tr_loader, va_loader, device):
        st = time.time()

        # define object
        save_dict = {}
        save_dict['tr_loss'] = []
        best_loss = 1000000000

        # optimizer #adam, outside loop, default lr
        opt = optim.Adam(self.model.parameters(), lr=self.lr)  # 1e-5
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.5, last_epoch=-1)

        for e in range(1, self.epoch + 1):
            loss_total = 0
            self.model.train()
            if e % 5 == 0:
                scheduler.step()
            print('\n==> Training Epoch #%d' % (e))
            for param_group in opt.param_groups:
                print("lr: ", param_group['lr'])

            # Training
            loss_train = 0
            for batch_idx, _input in enumerate(tr_loader):
                self.model.zero_grad()

                spec, div, labels = Variable(_input[0]).to(device), Variable(_input[1]).to(device), Variable(_input[2]).to(device)

                outputs = self.model(spec, div)

                loss = self.criterion(outputs, labels)
                loss.backward()
                opt.step()
                loss_train += loss.item()
            loss_train = loss_train / len(tr_loader)

            #print(spec, labels)
            #print(outputs)
            #v = input('train...')

            # Validate
            loss_val = 0
            for batch_idx, _input in enumerate(va_loader):
                #v = input('press key to continue...')
                spec, div, labels = Variable(_input[0]).to(device), Variable(_input[1]).to(device), Variable(_input[2]).to(device)

                # inputs.reshape(-1, inputs.shape[-1])
                outputs = self.model(spec, div)

                # calculate loss
                loss_val += self.criterion(outputs, labels).item()

            loss_val = loss_val / len(va_loader)
            #print(spec, labels)
            #print(outputs)
            #v = input('val...')

            # print model result
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Loss_train %4f  Loss_val %4f  Time %d'
                             % (e, self.epoch, loss_train, loss_val, time.time() - st))
            sys.stdout.flush()
            print('\n')

            # log data for visualization later
            log_value('train_loss', loss_train, e)
            log_value('val_loss', loss_val, e)

            # save model
            if loss_val < best_loss:
                save_dict['state_dict'] = self.model.state_dict()
                torch.save(save_dict, self.save_fn + 'model')
                best_epoch = e
                best_loss = loss_val

            # early stopping
            if (e - best_epoch) > 10:
                print(e, best_epoch)
                print('early stopping')
                break