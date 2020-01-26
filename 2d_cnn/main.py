from __future__ import print_function
from dataset import MyDataset
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as sio
from torchvision import datasets, transforms
from model_1d import PPNet
from model_vgg import vgg11
print("5 3 3 3")
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    params = list(model.named_parameters())
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().cuda(), target.type(torch.LongTensor).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(params[0][0], params[0][1].detach(), params[0][1].grad)

        if batch_idx % args.log_interval == 0:
            # for name, param in params:
            #     if(name == 'propagate1.alpha'):
            #         print('[{}]::[{}, {}]::[{}, {}]'.format(name, torch.min(param.detach()),\
            #             torch.max(param.detach()), torch.min(param.grad), torch.max(param.grad)))
            #     if(name == 'propagate2.alpha'):
            #         print('[{}]::[{}, {}]::[{}, {}]'.format(name, torch.min(param.detach()),\
            #             torch.max(param.detach()), torch.min(param.grad), torch.max(param.grad)))

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    cost = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().cuda(), target.type(torch.LongTensor).cuda()
            st = time.time()
            output = model(data)
            et = time.time()
            cost += (et - st)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(cost*1.0 / len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def adjust_learning_rate(optimizer, epoch):
    if(epoch % 20 == 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def main():
    # dog283
    NN_root = '/data/zj/local_pyproject/CNN/model/tf-idf/dog_283/features_NnN_900/features_NN_600/'
    NonrNN_root = '/data/zj/local_pyproject/CNN/model/tf-idf/dog_283/features_NnN_900/features_NoneNN_300/'
    train_data = '/data/zj/local_pyproject/CNN/data/train.txt'
    test_data = '/data/zj/local_pyproject/CNN/data/test.txt'
    val_data = '/data/zj/local_pyproject/CNN/data/val.txt'

    # bird 71
    # NN_root = '/data/zj/local_pyproject/CNN/model/tf-idf/bird71/features_NnN_900/features_NN_600/'
    # NonrNN_root = '/data/zj/local_pyproject/CNN/model/tf-idf/bird71/features_NnN_900/features_NoneNN_300/'
    # train_data = '/data/zj/local_pyproject/CNN/model/bird_71/partion/train.txt'
    # test_data = '/data/zj/local_pyproject/CNN/model/bird_71/partion/test.txt'
    # val_data = '/data/zj/local_pyproject/CNN/model/bird_71/partion/val.txt'


    train_set = MyDataset(NN_root, NonrNN_root, train_data)
    test_set = MyDataset(NN_root, NonrNN_root, test_data)
    val_set = MyDataset(NN_root, NonrNN_root, val_data)
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', # 16
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=64, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', # 0.5
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sp', default='saved_model',
                        help='model parameters path')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # device = torch.device("cuda:2" if use_cuda else "cpu")
    device = ""
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    data = sio.loadmat('../cal_similarity/similarity.mat')
    similarity = data['similarity']
    similarity = torch.from_numpy(similarity).float().cuda()

    # data = sio.loadmat('../dog_129/dog_wordAxis_900_sp.mat')
    data = sio.loadmat('/data/zj/re_generate_matrix_32691/validation_svm/axis/wordAxis_900.mat')
    NN_wv = data['NN_wv'].T
    NoneNN_wv = data['NoneNN_wv'].T
    NN_wv = torch.from_numpy(NN_wv).cuda()
    NoneNN_wv = torch.from_numpy(NoneNN_wv).cuda()

    model = PPNet(NN_wv, NoneNN_wv, similarity).cuda()
    # model = vgg11().cuda()

    print(model)

    # propagate_params = list(map(id, model.propagate.parameters()))
    # base_params = filter(lambda p: id(p) not in propagate_params, model.parameters())

    # optimizer = optim.SGD([
    #     {'params': model.propagate1.parameters(), 'lr': args.lr},
    #     {'params': model.propagate2.parameters(), 'lr':args.lr},
    #     {'params': model.conv1.parameters(), 'lr': args.lr},
    #     {'params': model.conv2.parameters(), 'lr': args.lr},
    #     {'params': model.fc1.parameters(), 'lr': args.lr},
    #     {'params': model.fc2.parameters(), 'lr': args.lr}
    #     ], momentum=args.momentum)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, val_loader)

    # torch.save(model.state_dict(), '../model_p_params/params_lr_0.005.pth')

    # params = list(model.named_parameters())
    # for name, param in params:
    #     if(name == 'propagate1.alpha'):
    #         print('{}::{}'.format(name+'1', param.detach()))
    #     if(name == 'propagate2.alpha'):
    #         print('{}::{}'.format(name+'2', param.detach()))
    test(args, model, device, test_loader)



if __name__ == '__main__':
    main()
