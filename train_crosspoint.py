# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
import argparse
import os

import numpy as np
import torch
import wandb
from lightly.loss import NTXentLoss
from sklearn.svm import SVC
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from CrossPoint.DGCNN_pc import DGCNN, DGCNN_partseg
from CrossPoint.Resnet_img import ResNet
from CrossPoint.data import ShapeNetRender, ModelNet40SVM
from CrossPoint.util import AverageMeter, IOStream


def train(args,io):
    wandb.init(project="CrossPoint",name=args.exp_name)
    #RandomHorizontalFlip 随机水平翻转。这意味着图像有50%的概率被水平翻转（即沿垂直轴翻转），而另外50%的概率保持原样。
    transform=transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                  ])
    train_loader=DataLoader(ShapeNetRender(img_transform=transform,n_imgs=2),num_workers=0,batch_size=args.batch_size,
                            shuffle=True,drop_last=True)
    device=torch.device("cuda" if args.cuda else "cpu")

    if args.model=='dgcnn':
        point_model=DGCNN(args).to(device)
    elif args.model=='dgcnn_seg':
        point_model=DGCNN_partseg(args).to(device)
    else:
        raise Exception("Not Implemented")

    img_model=ResNet(resnet50(),feat_dim=2048)
    img_model=img_model.to(device)

    wandb.watch(point_model)

    model = DGCNN(args).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.model_path))
        print("Model Loaded !!")

    parameters = list(point_model.parameters()) + list(img_model.parameters())

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,T_max=args.epochs,eta_min=0,last_epoch=-1)
    criterion=NTXentLoss(temperature=0.1).to(device)
    #在对比学习中，温度参数用于控制相似样本之间的相似度的敏感度。较低的温度值意味着模型需要更精确地区分相似和不相似的样本。

    best_acc=0
    for epoch in range(args.start_epoch,args.epochs):
        lr_scheduler.step()
        #Train
        train_losses=AverageMeter()
        train_imid_losses=AverageMeter()
        train_cmid_losses=AverageMeter()
        #同模态和跨模态使用了一样的损失函数

        point_model.train()
        img_model.train()
        wandb_log={}
        print(f'Start training epoch:({epoch}/{args.epochs})')
        for i,((data_t1,data_t2),imgs) in enumerate(train_loader):
            data_t1,data_t2,imgs=data_t1.to(device),data_t2.to(device),imgs.to(device)
            batch_size=data_t1.size()[0]

            opt.zero_grad()
            data=torch.cat(data_t1,data_t2)
            data=data.transpose(2,1).contiguous()
            _,point_feats,_=point_model(data)
            imgs_feats=img_model(imgs)

            point_t1_feats=point_feats[:batch_size,:]   #取前一半b
            point_t2_feats=point_feats[batch_size:,:]   #取后一半b

            loss_imid=criterion(point_t1_feats,point_t2_feats)
            point_feats=torch.stack([point_t1_feats,point_t2_feats]).mean(dim=0)
            #torch.stack函数接受一个张量列表，并沿着一个新的维度将它们堆叠起来。
            loss_cimd=criterion(point_feats,imgs_feats)

            total_loss=loss_cimd+loss_imid
            total_loss.backward()
            opt.step()

            train_losses.update(total_loss.item(),n=batch_size)
            train_cmid_losses.update(loss_cimd.item(),batch_size)
            train_imid_losses.update(loss_imid.item(),batch_size)

            if i % args.print_freq == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f, cmid loss: %.6f ' % (
                epoch, i, len(train_loader), train_losses.avg, train_imid_losses.avg, train_cmid_losses.avg))

        wandb_log['Train Loss'] = train_losses.avg
        wandb_log['Train IMID Loss'] = train_imid_losses.avg
        wandb_log['Train CMID Loss'] = train_cmid_losses.avg

        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)

        # Testing

        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=128, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=128, shuffle=True)

        feats_train = []
        labels_train = []
        point_model.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(device)
            with torch.no_grad():
                feats = point_model(data)[2]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        model_tl = SVC(C=0.1, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                               'img_model_best.pth')
            torch.save(img_model.state_dict(), save_img_model_file)

        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)

        wandb.log(wandb_log)

    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(point_model.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                       'img_model_last.pth')
    torch.save(img_model.state_dict(), save_img_model_file)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    args = parser.parse_args()

    #_init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)


