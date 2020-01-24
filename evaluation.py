import sys
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import model
from torch.autograd import Variable
from tqdm import tqdm
from utils import batch_accuracy, Tracker
from dataset import get_loader
import config


def evaluate_hw3():

    cudnn.benchmark = True

    train_loader = get_loader(train=True)
    val_loader = get_loader(val=True)

    net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
    state = torch.load('model.pth', map_location=lambda storage, loc: storage)
    weights = state["weights"]
    net.load_state_dict(weights)
    net.eval()

    tracker = Tracker()

    """ Run over the given loader """
    tracker_class, tracker_params = tracker.MeanMonitor, {}
    answ = []
    idxs = []
    accs = []

    tq = tqdm(val_loader, desc='Validation', ncols=0)
    loss_tracker = tracker.track('Validation_loss', tracker_class(**tracker_params))
    acc_tracker = tracker.track('Validation_acc', tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax().cuda()
    with torch.no_grad():
        for v, q, a, idx, q_len in tq:
            var_params = {
                'volatile': True,
                'requires_grad': False,
            }
            v = Variable(v.cuda(async=True), **var_params)
            q = Variable(q.cuda(async=True), **var_params)
            a = Variable(a.cuda(async=True), **var_params)
            q_len = Variable(q_len.cuda(async=True), **var_params)

            out = net(v, q, q_len)
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            acc = batch_accuracy(out.data, a.data).cpu()

            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

            # loss_tracker.append(loss.data[0])
            loss_tracker.append(loss.data)

            # acc_tracker.append(acc.mean())
            for a in acc:
                acc_tracker.append(a.item())
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))


evaluate_hw3()
