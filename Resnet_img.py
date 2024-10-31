# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
from torch import nn


class ResNet(nn.Module):
    def __init__(self,model,feat_dim=2048):
        super(ResNet, self).__init__()
        self.resnet=model
        self.resnet.fc=nn.Identity()
        self.projection_head=nn.Sequential(
            nn.Linear(in_features=feat_dim,out_features=512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,out_features=256,bias=False)
        )

    def forward(self,x):
        x=self.resnet(x)
        x=self.projection_head(x)
        return x
