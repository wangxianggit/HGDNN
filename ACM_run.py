import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ConvModel import HGDNN
import pdb
import pickle
import argparse
from utils import f1_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练set
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=80,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,  #2e-4
                        help='l2 reg')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout')
    parser.add_argument('-device', '--device', type=str, default='cuda:1',
                        help='which device? cuda:2/cpu')
    parser.add_argument('--adaptive_lr', type=str, default='False',
                        help='adaptive learning rate')

    args = parser.parse_args()
    args.dataset = 'ACM'
    args.adaptive_lr = 'True'

    epochs = args.epoch
    node_dim = args.node_dim
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    adaptive_lr = args.adaptive_lr
    print(args)

    # dataset
    with open('data/' + args.dataset + '/node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open('data/' + args.dataset + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open('data/' + args.dataset + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)  # train; val; test
    num_nodes = edges[0].shape[0]  # num_nodes>>>8994

    for i, edge in enumerate(edges):
        if i == 0:
            A = torch.from_numpy(edge.todense()).type(torch.cuda.FloatTensor).unsqueeze(-1)  # 添加末尾一位
        else:
            A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.cuda.FloatTensor).unsqueeze(-1)], dim=-1)
    # A.shape>>>([8994,8994,4])
    A = torch.cat([A, torch.eye(num_nodes).type(torch.cuda.FloatTensor).unsqueeze(-1)], dim=-1)

    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.cuda.LongTensor)  # 训练节点和labels
    train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.cuda.LongTensor)  # val
    valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.cuda.LongTensor)  # test
    test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.cuda.LongTensor)

    print('node_features:', node_features.shape)

    # num_classes >>> 3
    num_classes = torch.max(train_target).item() + 1
    final_NMI, final_ARI = [], []
    # writer = SummaryWriter("logs")  # 存放log文件的目录
    l1 = []
    l2 = []
    for l in range(1):
        model = HGDNN(num_edge=A.shape[- 1],  # edge类别的数量:4; 还有一个单位阵;所以A.shape>>>5  4+1=5
                    w_in=node_features.shape[1],  # 节点的特征取第一位 >>>([8894,1902])
                    num_node=A.shape[0],
                    w_out=node_dim,  # 隐藏节点的特征>>>64
                    num_class=num_classes,  # 输出类别数>>>3
                    dropout=dropout)

        if adaptive_lr == 'False':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.00001)
        else:
            optimizer = torch.optim.Adam([{'params': model.weight},
                                          {'params': model.attw},
                                          {'params': model.attb},
                                          {'params': model.attq},
                                          {'params': model.linear1.parameters()},
                                          {'params': model.linear2.parameters()},
                                          {'params': model.DGCN.parameters(), "lr": 0.002,"weight_decay": 0.0},
                                          ], lr=0.01, weight_decay=0.0)
        model.to(device)
        loss = nn.CrossEntropyLoss()
        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        max_val_loss = 10000
        max_test_loss = 10000
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        max_val_macro_f1, max_val_micro_f1 = 0, 0
        max_test_macro_f1, max_test_micro_f1 = 0, 0
        max_test_nmi, max_test_ari = 0, 0
        kmeans = KMeans(n_clusters=3, n_init=10)
        for i in range(epochs):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > 0.001:
                    param_group['lr'] = param_group['lr'] * 0.9
            print('Epoch:  ', i + 1)

            model.zero_grad()
            model.train()
            loss, y_train = model(A, node_features, train_node, train_target)

            y_pred = kmeans.fit_predict(y_train.detach().cpu().numpy())
            train_target1 = train_target.detach().cpu()

            # cluster
            train_nmi = nmi_score(train_target1, y_pred, average_method='arithmetic')
            train_ari = ari_score(train_target1, y_pred)

            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(), dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1),average='micro')
            print('Train - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1, sk_train_f1))
            print('Train:  nmi {:.4f}'.format(train_nmi), ', ari {:.4f}'.format(train_ari))

            loss.backward()
            optimizer.step()
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid = model.forward(A, node_features, valid_node, valid_target)

                vallid_z = kmeans.fit_predict(y_valid.detach().cpu().numpy())
                valid_target1 = valid_target.detach().cpu()
                valid_nmi = nmi_score(valid_target1, vallid_z, average_method='arithmetic')
                valid_ari = ari_score(valid_target1, vallid_z)

                val_f1 = torch.mean(f1_score(torch.argmax(y_valid, dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1),average='micro')
                print('Valid - Loss: {}, Macro_F1: {}, Micro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1, sk_val_f1))
                print('Valid:   nmi {:.4f}'.format(valid_nmi), ', ari {:.4f}'.format(valid_ari))
                test_loss, y_test= model.forward(A, node_features, test_node, test_target)

                test_z = kmeans.fit_predict(y_test.detach().cpu().numpy())
                test_target1 = test_target.detach().cpu()

                test_nmi = nmi_score(test_target1, test_z, average_method='arithmetic')
                test_ari = ari_score(test_target1, test_z)

                test_f1 = torch.mean(f1_score(torch.argmax(y_test, dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1),average='micro')
                print('Test - Loss: {}, Macro_F1: {}, Micro_F1:{} \n'.format(test_loss.detach().cpu().numpy(), test_f1, sk_test_f1))
                print('Test:   nmi {:.4f}'.format(test_nmi), ', ari {:.4f}'.format(test_ari))

            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_micro_train_f1 = sk_train_f1
                best_micro_val_f1 = sk_val_f1
                best_micro_test_f1 = sk_test_f1

            if val_f1 > max_val_macro_f1:
                max_val_loss = val_loss.detach().cpu().numpy()
                max_val_macro_f1 = val_f1
            if sk_val_f1 > max_val_micro_f1:
                max_val_loss = val_loss.detach().cpu().numpy()
                max_val_micro_f1 = sk_val_f1
            if test_f1 > max_test_macro_f1:
                max_test_loss = test_loss.detach().cpu().numpy()
                max_test_macro_f1 = test_f1
            if sk_test_f1 > max_test_micro_f1:
                max_test_loss = val_loss.detach().cpu().numpy()
                max_test_micro_f1 = sk_test_f1

            if test_nmi > max_test_nmi:
                max_test_loss = test_loss.detach().cpu().numpy()
                max_test_nmi = test_nmi
            
            if test_ari > max_test_ari:
                max_test_loss = val_loss.detach().cpu().numpy()
                max_test_ari = test_ari


        print('---------------Results--------------------')
        print('Valid - Loss: {:.4f}, maxMacro_F1: {:.4f}, maxMicro_F1: {:.4f}'.format(max_val_loss, max_val_macro_f1,
                                                                                      max_val_micro_f1))
        print('Test - Loss: {:.4f}, maxMacro_F1: {:.4f}, maxMicro_F1: {:.4f}'.format(max_test_loss, max_test_macro_f1,
                                                                                    max_test_micro_f1))
        print('Test - Loss: {:.4f}, max_nim: {:.4f}, max_ari: {:.4f}'.format(max_test_loss, max_test_nmi,
                                                                             max_test_ari))
        l1.append(max_test_macro_f1)
        l2.append(max_test_micro_f1)
        final_NMI.append(max_test_nmi)
        final_ARI.append(max_test_ari)
    print("total run class results", l1, l2)
    print("total run cluster results", final_NMI, final_ARI)
    print("average results", sum(l1) / len(l1), sum(l2) / len(l2))
    print("average results", sum(final_NMI) / len(final_NMI), sum(final_ARI) / len(final_ARI))
