from dataloader import train_loader
from torch.autograd import Variable
import torch.optim as optim
from params import *
from utils import *
from model import classifier
import torch.nn.functional as F
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.5)

def train(epoch,dataloader):
    loss = []
    f1score = []
    for i in range(epoch):
        y_pred = []
        y_true = []
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = Variable(data), Variable(target)
            data = data.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            output = classifier(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            output = output.cpu().detach().numpy()
            y_pred.append(output)
            target = target.cpu().numpy()
            y_true.append(target)
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data), len(dataloader.dataset),100. * batch_idx / len(dataloader), loss.item()))
        y_pred = get_pred(y_pred)
        f_score = get_fscore(y_true,y_pred)
        loss.append(loss.item)
        f1score.append(f_score)
        print('Train Epoch: {} \tf1_score: {:.6f}'.format(epoch , f_score))
    return loss,f1score

#return list of loss and f1score per Epoch
loss,f1score = train(epoch,train_loader)
