import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = models.resnet50(pretrained = True)
        self.num_ftrs = self.resnet.fc.in_features
        self.l1 = nn.Linear(1000 , 256)
        self.l2 = nn.Linear(256,17)
    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = F.relu(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)
#uncomment to see summary of the created model
#summary(classifier,(3,224,224))
