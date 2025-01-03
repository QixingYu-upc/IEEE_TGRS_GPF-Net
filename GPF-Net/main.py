import numpy as np
import torch
import time
import sys
import data_reader
import utils
import sys
import scipy.io as scio
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from GNN import split_data_graph as split_data
from GNN import create_graph
from GNN import dr_slic
import AF2GNN
import gc
# import Data_save
import os
import random
from CEGCN import CEGCN
from WFCG import WFCG
np.set_printoptions(threshold=np.inf)
import warnings
warnings.filterwarnings("ignore")
def seed_torch(seed=128,deter=False):
    '''
    `deter` means use deterministic algorithms for GPU training reproducibility, 
    if set `deter=True`, please set the environment variable `CUBLAS_WORKSPACE_CONFIG` in advance
    '''
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    # torch.set_deterministic(deter)  # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(deter)

# seed_torch()
def load_data():
    data = data_reader.IndianRaw().normal_cube
    data_gt = data_reader.IndianRaw().truth
    data_gt = data_gt.astype('int')
    return data,data_gt
data, data_gt = load_data()
class_num = np.max(data_gt)
gt_reshape = np.reshape(data_gt, [-1])
samples_type = ['ratio','same_num'][0]
train_ratio = 0.001
val_ratio = 0.001 
train_num = 10
val_num = class_num
learning_rate = 2e-3
max_epoch = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
superpixel_scale = 100
dataset_name = "indian_"
path_model = r"model1\\"
path_data = None
height,width,bands = data.shape
A = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/A.mat')['A']
Q = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/Q.mat')['Q']
train_samples_gt = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/train_samples_gt.mat')['train_samples_gt']
test_samples_gt = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/test_samples_gt.mat')['test_samples_gt']
val_samples_gt = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/val_samples_gt.mat')['val_samples_gt']

train_samples_gt_onehot = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/train_samples_gt_onehot.mat')['train_samples_gt_onehot']
test_samples_gt_onehot = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/test_samples_gt_onehot.mat')['test_samples_gt_onehot']
val_samples_gt_onehot = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/val_samples_gt_onehot.mat')['val_samples_gt_onehot']

train_label_mask = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/train_label_mask.mat')['train_label_mask']
test_label_mask = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/test_label_mask.mat')['test_label_mask']
val_label_mask = scio.loadmat('D:\\yqx\\Practice\\NL_GNN_for_HSI\\Datasets\\Indian Pines/val_label_mask.mat')['val_label_mask']
# print(A.shape)
Q = torch.from_numpy(Q).to(device)
A = torch.from_numpy(A).to(device)

train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

net_input = np.array(data,np.float32)
net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

#训练
for k in range(10):
    print('--------------',k)#Salinas原来是150
    net = AF2GNN.Fusion_GNN_CNN(in_channels=bands,nhid=60,out_channels=64,dropout=0.5,alpha=0.2,nheads=8,filter_num=2,class_num=class_num,Q=Q).to(device)
    # net = AF2GNN.Fusion_GNN_CNN(in_channels=200,nhid=100,out_channels=64,dropout=0.5,alpha=0.2,nheads=8,filter_num=2,class_num=16,Q=Q).to(device)#indian_lr = 3e-3
    # net = AF2GNN.Fusion_GNN_CNN(in_channels=103,nhid=100,out_channels=64,dropout=0.5,alpha=0.2,nheads=8,filter_num=2,class_num=9,Q=Q).to(device)#PaviaU_lr = 3e-3
    # net = AF2GNN.Fusion_GNN_CNN(in_channels=204,nhid=150,out_channels=64,dropout=0.5,alpha=0.2,nheads=8,filter_num=2,class_num=16,Q=Q).to(device)#Salinas_lr = 3e-3
    # net = AF2GNN.Fusion_GNN_CNN(in_channels=270,nhid=60,out_channels=64,dropout=0.5,alpha=0.2,nheads=8,filter_num=2,class_num=9,Q=Q).to(device)#WHU_lr=2e-3
    # net = Data_save.GCN(in_channels=bands,in_hid=200,classes=64,out = class_num,Q=Q).to(device)
    # net = CEGCN(height=height,width=width,changel=bands,class_count=class_num,Q=Q,A=A).to(device)
    # net = WFCG(height=height,width=width,changel=bands,class_count=class_num,Q=Q,A=A).to(device)
    # net = Data_save.Res_GCN(in_channels=200,out_channels=64,nhid=100,Q=Q).to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-3)
    zeros = torch.zeros([height*width]).to(device).float()
    best_loss = 99999
    net.train()
    tic1 = time.time()
    for i in range(max_epoch + 1):
        optimizer.zero_grad()
        output = net(net_input,A)
        loss = utils.compute_loss(output,train_samples_gt_onehot,train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()

        with torch.no_grad():
            net.eval()
            output = net(net_input,A)
            print(output.shape)
            trainloss = utils.compute_loss(output,train_samples_gt_onehot,train_label_mask)
            trainOA = utils.evaluate_performance(output, train_samples_gt, train_samples_gt_onehot, zeros)
            valloss = utils.compute_loss(output, val_samples_gt_onehot, val_label_mask)
            valOA = utils.evaluate_performance(output, val_samples_gt, val_samples_gt_onehot, zeros)
            # torch.save(net.state_dict(),path_model + r"model.pt")
            if valloss < best_loss:
                best_loss = valloss
                torch.save(net.state_dict(),path_model + r"model.pt")
        torch.cuda.empty_cache()
        net.train()

        if i%20==0:
            print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss,trainOA, valloss, valOA))
    toc1 = time.time()
    print("\n\n====================training done. starting evaluation...========================\n")
    torch.cuda.empty_cache()
    with torch.no_grad():
        zeros = torch.zeros([height * width]).to(device).float()
        net.load_state_dict(torch.load(path_model + r"model.pt"))
        net.eval()
        tic2 = time.time()
        output = net(net_input,A)
        output = torch.squeeze(output)
        toc2 = time.time()
        testloss = utils.compute_loss(output,test_samples_gt_onehot,test_label_mask)
        testOA = utils.evaluate_performance(output,test_samples_gt,test_samples_gt_onehot,zeros)
        print("{}\ttest loss={:.4f}\t test OA={:.4f}".format(str(i + 1), testloss, testOA))
    torch.cuda.empty_cache()
    train_time = toc1 - tic1
    test_time = toc2 - tic2

    print("Train time :%d"%train_time)
    print("Test time :%d"%test_time)
    test_label_mask_cpu = test_label_mask.cpu().numpy()[:,0].astype('bool')
    test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64').T
    predict = torch.argmax(output,dim=1).cpu().numpy()

    classfication = classification_report(test_samples_gt_cpu[test_label_mask_cpu],predict[test_label_mask_cpu]+1,digits=4)
    kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu],predict[test_label_mask_cpu]+1)
    print(classfication)
    print("kappa",kappa)
    # net.load_state_dict(torch.load(path_model + r"model.pt"))
    # # utils.Draw_Classification_Map(net,net_input,A,device,'/home/project/GNN_for_HSI/NL_GNN_for_HSI/photos/11')
    _,total_indices = utils.sampling(1,data_gt)
    # # print(len(total_indices))
    utils.generate_png(net,net_input,data_gt,device,total_indices,'/home/project/GNN_for_HSI/NL_GNN_for_HSI/photos/Indian_pines',A)
    del net