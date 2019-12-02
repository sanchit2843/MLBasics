from torch import nn
import sys
import adabound
from torch.autograd import Variable
from lib.Earlystopping import EarlyStopping
from lib.lr_finder import LRFinder
from lib.onecycle import OneCycle
from lib.util import *

#Lr update for cyclical learning rate
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom
        

def train(model,dataloaders,device,num_epochs,lr,batch_size,patience):
    i = 0
    phase1 = dataloaders.keys()
    losses = list()
    criterion = nn.CrossEntropyLoss()
    acc_all = list()
    best_acc = 0
    optimizer = adabound.AdaBound(model.parameters(), lr=lr)
    onecyc = OneCycle(len(dataloaders['train'])*num_epochs,lr)
    if(patience!=None):
        earlystop = EarlyStopping(patience = patience,verbose = True)
    for epoch in range(num_epochs):
        print('Epoch:',epoch)
        epoch_metrics = {"loss": [], "acc": []}

        for phase in phase1:
            if phase == ' train':
                model.train()
            else:
                model.eval()
            for  batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.FloatTensor).to(device)
                target = target.type(torch.LongTensor).to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                acc = 100 * (output.detach().argmax(1) == target).cpu().numpy().mean()
                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["acc"].append(acc)
                sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    epoch,
                    num_epochs,
                    batch_idx,
                    len(dataloaders[phase]),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    )
                )

                if(phase =='train'):
                    loss.backward()
                    optimizer.step()
                    lr,_ = onecyc.calc()
                    update_lr(optimizer, lr)
                    #update_mom(optimizer, mom)
                
            epoch_acc = np.mean(epoch_metrics["acc"])
            epoch_loss = np.mean(epoch_metrics["loss"])
            #if(epoch_acc>best_acc):
              #best_acc = epoch_acc
              #torch.save(model.state_dict(),'/content/drive/My Drive/weights_genre/dense121.h5')
            if(phase == 'val' and patience !=None):
                earlystop(epoch_loss,model)
                if(earlystop.early_stop):
                    print("Early stopping")
                    model.load_state_dict(torch.load('./checkpoint.pt'))
                    print('{} Accuracy: {}'.format(phase,epoch_acc.item()))
                    break
            if(phase == 'train'):
                losses.append(epoch_loss)
                acc_all.append(epoch_acc)

        print('{} Accuracy: {}'.format(phase,epoch_acc.item()))
    return losses,acc_all
def test(model,dataloader,device,batch_size):
    running_corrects = 0
    running_loss=0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim = 1)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.FloatTensor).to(device)
        target = target.type(torch.LongTensor).to(device)
        output = model(data)
        loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds,(len(preds),1))
        target = np.reshape(target,(len(preds),1))
        data = data.cpu().numpy()

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if(preds[i]!=target[i]):
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    epoch_acc = running_corrects.double()/(len(dataloader)*batch_size)
    epoch_loss = running_loss/(len(dataloader)*batch_size)
    print(epoch_acc,epoch_loss)
    return true,pred,image,true_wrong,pred_wrong


#Complete code with inference
def train_model(model,dataloaders,encoder,lr_scheduler = None,inv_normalize = None,num_epochs=10,lr=0.0001,batch_size=8,patience = None,classes = None):
    dataloader_train = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = dataloaders.keys()
    perform_test = False
    for phase in key:
        if(phase == 'test'):
            perform_test = True
        else:
            dataloader_train.update([(phase,dataloaders[phase])])
    losses,accuracy = train(model,dataloader_train,device,num_epochs,lr,batch_size,patience)
    error_plot(losses)
    acc_plot(accuracy)
    if(perform_test == True):
        true,pred,image,true_wrong,pred_wrong = test(model,dataloaders['test'],device,batch_size)
        wrong_plot(12,true_wrong,image,pred_wrong,encoder,inv_normalize)
        performance_matrix(true,pred)
        if(classes !=None):
            plot_confusion_matrix(true, pred, classes= classes,title='Confusion matrix, without normalization')
    return losses,accuracy
  
  
