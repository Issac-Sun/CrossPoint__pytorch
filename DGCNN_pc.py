# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

def knn(x,k):
    inner=-2*torch.matmul(input=x.transpose(2,1),other=x)
    xx=torch.sum(x**2,dim=1,keepdim=True)
    pairwise_distance=-xx-inner-xx.transpose(2,1)
    idx=pairwise_distance.topk(k=k,dim=-1)[1]
    # topk函数用于从给定的张量中选择最大的k个元素及其索引。在这个上下文中，你需要找到每个点的k个最近邻点，这意味着你需要找到距离最小的k个点。
    # pairwise_distance.topk(k=k, dim=-1)：这个调用返回每个点的k个最近小距离及其对应的索引。
    # dim=-1指定了在最后一个维度（即每个点的所有距离）上进行操作。
    # topk函数返回两个张量：一个是值（最大的k个距离），另一个是索引（这些距离对应的点的索引）。
    return idx

def get_graph_feature(x,k=20,idx=None):
    batch_size=x.size(0)
    num_points=x.size(2)
    x=x.view(batch_size,-1,num_points)
    if idx is None:
        idx=knn(x=x,k=k)
    device=torch.device('cuda:1')
    idx_base=torch.arange(start=0,end=batch_size,step=1,device=device).view(-1,1,1)*num_points
    #创建一个全局索引
    idx=idx+idx_base
    idx=idx.view(-1)
    _,num_dims,_=x.size()
    x=x.transpose(2,1).contiguous()
    feature=x.view(batch_size*num_points,-1)[idx,:]
    feature=feature.view(batch_size,num_points,k,num_dims)
    x=x.view(batch_size,num_points,1,num_dims).repeat(1,1,k,1)
    # 创建一个包含每个点特征的三维张量，其中每个点的特征被复制 k 次
    # 以便于与每个点的 k 个最近邻的特征进行操作。
    # 结果是一个形状为 (batch_size, num_points, k, num_dims) 的张量。
    feature=torch.cat((feature-x,x),dim=3).permute(0,3,1,2).contiguous()
# feature - x 的形状是 (batch_size, num_points, k, num_dims)。
# x 的形状是 (batch_size, num_points, k, num_dims)
    return feature

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN(nn.Module):
    def __init__(self, args, cls=-1):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        if cls != -1:
            self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=args.dropout)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=args.dropout)
            self.linear3 = nn.Linear(256, output_channels)

        self.cls = cls

        self.inv_head = nn.Sequential(
            nn.Linear(args.emb_dims * 2, args.emb_dims),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(inplace=True),
            nn.Linear(args.emb_dims, 256)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        feat = x
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)

        inv_feat = self.inv_head(feat)

        return x, inv_feat, feat


class DGCNN_partseg(nn.Module):
    def __init__(self,args,seg_num_all=None,pretrain=True):
        super(DGCNN_partseg, self).__init__()
        self.args=args
        self.seg_num_all=seg_num_all
        self.k=args.k
        self.pretrain=pretrain
        self.transform_net=Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.projection_head=nn.Sequential(
            nn.Linear(in_features=args.emb_dims,out_features=args.emb_dims),
            nn.BatchNorm1d(args.emb_dims),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=args.emb_dims,out_features=256)

        )
        #判断是否有pretrain这里，如果没有就是完整的一个做分类的DGCNN了（if not下面一大坨），
        # 如果有就会截至在做完max pooling这，然后进入projection_head，不用走那么长了。
        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=args.dropout)
            self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       self.bn9,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp2 = nn.Dropout(p=args.dropout)
            self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn10,
                                        nn.LeakyReLU(negative_slope=0.2))
            self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self,x,label=None):
        batch_size=x.size(0)
        num_points=x.size(2)

        x0=get_graph_feature(x=x,k=self.k)
        t=Transform_Net(x0)
        x=x.transpose(2,1)      ## (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x=torch.bmm(x,t)         # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x=x.transpose(2,1)      #b,3,n

        x=get_graph_feature(x=x,k=self.k)
        x=self.conv1(x)
        x=self.conv2(x)
        x1=x.max(dim=-1,keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x=torch.cat((x1,x2,x3),dim=1)   #b,64*3,num_points
        x=self.conv6(x)
        x=x.max(dim=-1,keepdim=True)[0]

        if self.pretrain:
            print("----Pretrain-----")
            x=x.squeeze()
            #用于去除张量（tensor）中所有长度为1的维度。这个方法返回一个新的张量，它与原张量共享数据，但是没有长度为1的维度。
            # 如果原张量中没有长度为1的维度，则返回原张量的视图。
            projection_feature=self.projection_head(x)
            return x,projection_feature,x
        #这种返回多个值的做法在Python中是完全合法的，并且可以方便地将多个相关的结果一次性返回给调用者。
        # 调用者可以通过多个变量来接收这些返回值

        else:
            label=label.view(batch_size,-1,1)       #(b,num_cat,1)
            label=self.conv7(label)

            x=torch.cat((x,label),dim=1)        #拼接
            x=x.repeat(1,1,num_points)

            x=torch.cat((x,x1,x2,x3),dim=1)
            x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
            x = self.dp1(x)
            x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
            x = self.dp2(x)
            x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
            x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
            return x







