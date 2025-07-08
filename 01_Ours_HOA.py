import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import time
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
# from net_last_muti import ResNet
from torch.utils.data import TensorDataset
import random
from CrossPIR_fusion_add_loss_HOA import FusionModel
#from fusion_light import FusionModel
from HOA import Mydataset
import copy

def cal_acc(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    mean_acc = np.round(np.mean(ACC), 4)
    return mean_acc

softmax = nn.Softmax(dim=1)
tanh = nn.Tanh()
relu = nn.ReLU(inplace=False)
def setup_seed(seed, seed1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed1)
    random.seed(seed1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_prototype(model, dataloader, device, epoch, a_proto=None, v_proto=None):

    n_classes=4

    audio_prototypes = torch.zeros(n_classes, 256).to(device)
    visual_prototypes = torch.zeros(n_classes, 256).to(device)
    count_class = [0 for _ in range(n_classes)]#初始化原型和计数器

    # calculate prototype
    model.eval()#进入模型评估模式
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        for i, (angles, degree) in enumerate(train_loader, 1):

            inputs, labels = angles, degree
            inputs, labels = torch.from_numpy(np.array(inputs)).cuda(), torch.from_numpy(np.array(labels)).cuda()
            # TODO: make it simpler and easier to extend
            # outputs, outputs1, features1, features2, loss_ita = model(inputs.float())
            output_local, output_global, outputs, outputs1, features1, features2, loss_ita = model(inputs.float())

            v = features1
            a = features2


            for c, l in enumerate(labels):#遍历每个样本，将其特征累加到对应类别的原型向量中，并更新类别计数器
                # print(labels)
                l = l.long()
                count_class[l] += 1
                audio_prototypes[l, :] += a[c, :]   #用于存储每个类别的原型向量
                visual_prototypes[l, :] += v[c, :]


            sample_count += 1

            if sample_count >= all_num // 10:#仅使用总数据的10%来计算原型，增加计算效率
                break
    for c in range(audio_prototypes.shape[0]):
        audio_prototypes[c, :] /= count_class[c]
        visual_prototypes[c, :] /= count_class[c]# 将累加的特征向量除以类别样本数，得到每个类别的平均原型向量

    if epoch <= 0:
        audio_prototypes = audio_prototypes
        visual_prototypes = visual_prototypes
    else:
        audio_prototypes = (1 - 0.2) * audio_prototypes + 0.2 * a_proto
        visual_prototypes = (1 - 0.2) * visual_prototypes + 0.2 * v_proto #结合前一轮的原型向量进行更新，使用移动平均的方法。
    return audio_prototypes, visual_prototypes
def EU_dist(x1, x2):#计算欧氏距离
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix

if __name__ == '__main__':
    seedlist = [5021, 5021,3941, 3941]
    for index01 in range(4):
        start = time.time()
        seed0 = seedlist[index01]
        print("随机种子为：", seed0)
        print("循环到第几次:", index01)
        setup_seed(seed0, seed0)
        kern_size = [int(l) for l in "9,5,3".split(",")]
        print("start searching at ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))
        # num_rev_HC, num_rev_CSM, num_rev_PD = 1/33, 1/55, 1/45
        # criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([num_rev_HC, num_rev_CSM, num_rev_PD])).float())
        criterion = nn.CrossEntropyLoss()

        os.environ[
            "CUDA_VISIBLE_DEVICES"] = '0'#根据GPU改动，先看哪块没在用:watch -n 1 nvidia-smi
        device = "gpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        # device = torch.device("cuda:0")

        criterion = criterion.to(device)

        batch_size = 8
        num_classes = 4
        num_epoch = 150
        final_matrix = 0
        final_acc = {}
        final_confusion_matrix = {}

        X, Y, groups= Mydataset()
        print("数据加载成功")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        sgkf.get_n_splits(X, Y)
        print(sgkf)

        SKF = 1
        Loss1 = np.zeros((1, num_epoch))
        Loss1_val = np.zeros((1, num_epoch))
        acc_train1 = np.zeros((1, num_epoch))
        acc_val1 = np.zeros((1, num_epoch))
        sum1 = 0
        acc_per=[]

        for i, (train_index, test_index) in enumerate(sgkf.split(X, Y, groups)):
            print('------------------------------ skf = ', SKF, '------------------------------')
            SKF = SKF + 1
            val_losses02 = []
            acc02 = []
            train_loss02 = []

            accuracy_low_best = 0
            Val_ACC_best = [0, 0, 0]
            print(f"Fold{i:}")
            print(f"Train:  index={train_index}")
            print(f"        group={groups[train_index]}")
            # print(f"        sample={sample[train_index]}")
            # NOTE:打印出训练集对应的索引号
            print(f"Test:  index={test_index}")
            print(f"        group={groups[test_index]}")
            # print(f"        sample={sample[test_index]}")
            count0 = np.sum(Y[train_index] == 0)
            count1 = np.sum(Y[train_index] == 1)
            count2 = np.sum(Y[train_index] == 2)
            count3 = np.sum(Y[train_index] == 3)

            print('训练集中的1标签个数：', count0)
            print('训练集中的2标签个数：', count1)
            print('训练集中的3标签个数：', count2)
            print('训练集中的4标签个数：', count3)

            data_train = X[train_index]  # data_train(110,10818)
            # data_train = np.swapaxes(data_train, 1, 2)
            data_val = X[test_index]  # data_val(19,10818)
            # data_val = np.swapaxes(data_val, 1, 2)
            label_train = Y[train_index]  # lable_train(110,)
            label_val = Y[test_index]  # lable_val(19,)
            train_dataset = TensorDataset(torch.from_numpy(data_train), torch.from_numpy(label_train))
            val_dataset = TensorDataset(torch.from_numpy(data_val), torch.from_numpy(label_val))

            #增加初始样本加载权重为平均数
            sample_weight = []
            for ii in range(len(train_index)):
                sample_weight.append(1 / len(train_index))
            sample_weight = np.array(sample_weight)
            # print('初始的样本权重：', sample_weight)
            sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_index), replacement=True)

            # data_loader
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=8,
                                                       num_workers=0,
                                                       drop_last=False,
                                                       sampler=sampler)

            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=len(test_index),
                                                     shuffle=False,
                                                     num_workers=0,
                                                     drop_last=False)
            Loss = {}
            Epoch = {}
            acc_train = {}
            acc_val = {}
            acc_val_local = {}
            acc_val_global = {}
            Loss_val = {}

            best_acc = 0
            best_MATRIX = 0

            best_epoch = 0
            best_labels1 = np.zeros([1, 4])
            best_probability1 = np.zeros([1, 4])

            net = FusionModel(hidden_size1=256, hidden_size2=256, num_classes=4, num_dp=0.3)
            net = net.to(device)


            optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 70, 0.1)


            epoch=0
            audio_proto, visual_proto = calculate_prototype(net, train_loader, device, epoch)
            for epoch in range(num_epoch):
                s = epoch + 1
                running_loss = 0.0
                running_loss_val = 0.0
                MATRIX_train = 0
                MATRIX = 0
                MATRIX_local = 0
                MATRIX_global = 0
                len_input = 0
                len_input_val = 0
                patience = 20
                x_get_train_before = []
                early_stopping_count = 0

                for i, (angles, degree) in enumerate(train_loader, 1):

                    inputs, labels = angles, degree
                    net.train()

                    inputs = inputs

                    inputs, labels = torch.from_numpy(np.array(inputs)).cuda(), torch.from_numpy(np.array(labels)).cuda()

                    len_input = len_input + len(inputs)
                    output_local, output_global, outputs, outputs1, features1, features2, loss_ita = net(inputs)

                    # outputs, outputs1, features1, features2, loss_ita = net(inputs)
                    # x_get_train_before.extend(outputs_get.cpu().detach().numpy())

                    labels = labels.long()

                    # loss = criterion(outputs, labels)

                    optimizer.zero_grad()

                    # '/*********************************PMR**************************************/'
                    # TODO: make it simpler and easier to extend

                    clf = net.fusion_layer

                    cli = features1
                    img = features2

                    audio_sim = -EU_dist(cli, audio_proto)  # B x n_class计算该样本对应特征到原型之间的距离
                    visual_sim = -EU_dist(img, visual_proto)  # B x n_class
                    # print('sim: ', audio_sim[0][0].data, visual_sim[0][0].data, a[0][0].data, v[0][0].data)

                    if 0 <= epoch <= 100:

                        score_a_p = sum([softmax(audio_sim)[i][labels[i]] for i in range(audio_sim.size(0))])
                        score_v_p = sum([softmax(visual_sim)[i][labels[i]] for i in range(visual_sim.size(0))])
                        ratio_a_p = score_a_p / score_v_p

                        loss_proto_a = criterion(audio_sim, labels)
                        loss_proto_v = criterion(visual_sim, labels)
                        # print("增加的损失值",loss_proto_a, loss_proto_v, loss_ita)

                        if ratio_a_p > 1:
                            beta = 0  # audio coef
                            lam = 1 * 1.0  # visual coef
                        elif ratio_a_p < 1:
                            beta = 1 * 1.0
                            lam = 0
                        else:
                            beta = 0
                            lam = 0
                        loss = criterion(outputs, labels) + beta * loss_proto_a + lam * loss_proto_v + 0.03*loss_ita

                    else:
                        loss = criterion(outputs, labels) + 0.03*loss_ita


                    loss.backward()


                    # '/**********************************************************************************/'
                    optimizer.step()

                    running_loss += loss.item()

                    _, predicted = torch.max(outputs1, 1)
                    cnf_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1, 2, 3 ])
                    MATRIX_train = MATRIX_train + np.array(cnf_matrix)

                Loss[s] = running_loss / len_input
                Epoch[s] = epoch
                acc_train[s] = cal_acc(MATRIX_train)

                with torch.no_grad():
                    all_predict01 = []
                    x_get_test_before = []
                    # net_best = 0#由于可能最后一个epoch会出现准确率不如上一个的现象，则net_best归0
                    for i, (angles, degree) in enumerate(val_loader, 1):
                        inputs, labels = angles, degree

                        inputs = inputs




                        inputs, labels = torch.from_numpy(np.array(inputs)).cuda(), torch.from_numpy(np.array(labels)).cuda()

                        len_input_val = len_input_val + len(inputs)

                        net.eval()
                        # outputs, outputs1, features1, features2, loss_ita= net(inputs)
                        output_local, output_global, outputs, outputs1, features1, features2, loss_ita = net(inputs)

                        all_predict01.extend(outputs.cpu().numpy())
                        # x_get_test_before.extend(outputs_get.cpu().detach().numpy())

                        # print('第一个分类器的分类效果：',yo_train_pred_ori)

                        one_hot = torch.zeros(len(outputs), num_classes).long()
                        # labels1 = one_hot.scatter_(dim=1, index=labels.cpu().unsqueeze(dim=1),
                        #                            src=torch.ones(len(outputs), num_classes).long())

                        loss = criterion(outputs, labels.long()) + 0.03*loss_ita
                        running_loss_val += loss.item()

                        #以下为了打印出局部和全局分类器分类准确率
                        _, predicted = torch.max(outputs1, 1)
                        _, predicted_local = torch.max(output_local, 1)
                        _, predicted_global = torch.max(output_global, 1)
                        cnf_matrix_local = confusion_matrix(labels.cpu().numpy(), predicted_local.cpu().numpy(),
                                                            labels=[0, 1, 2, 3])
                        cnf_matrix_global = confusion_matrix(labels.cpu().numpy(), predicted_global.cpu().numpy(),
                                                             labels=[0, 1, 2, 3])
                        cnf_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1, 2, 3])
                        MATRIX = MATRIX + np.array(cnf_matrix)
                        MATRIX_local = MATRIX_local + np.array(cnf_matrix_local)
                        MATRIX_global = MATRIX_global + np.array(cnf_matrix_global)

                        acc_val[s] = cal_acc(MATRIX)
                        acc_val_local[s] = cal_acc(MATRIX_local)
                        acc_val_global[s] = cal_acc(MATRIX_global)

                        if np.array(acc_val[s]) >= best_acc:
                            best_acc = np.array(acc_val[s])
                            best_epoch = epoch
                            best_MATRIX = MATRIX
                            # print(net_best)
                            # best_labels = np.array(labels1)
                            best_net = copy.deepcopy(net)
                            best_probability = outputs1.cpu().numpy()

                            best_classes = labels.cpu().numpy()
                            best_predicted = predicted.cpu().numpy()

                        y_pred_ori_acc = best_acc


                    # print('第二个分类器的分类效果：',yo_train_pred_ori_2)
                    yo_train_pred_ori = np.array(all_predict01)

                # scheduler.step(cal_acc(MATRIX))
                Loss_val[s] = running_loss_val / len_input_val
                print('epoch:', epoch, '------',
                      'loss_train:', np.round(Loss[s], 4), '------',
                      'acc_train:', np.round(acc_train[s], 4), '------',
                      'loss_val:', np.round(Loss_val[s], 4), '------',
                      'acc_val:', np.round(acc_val[s], 4), '------')
            print('skf=', SKF, ' best_acc=', best_acc)
            sum1+=best_acc
            acc_per.append(best_acc)
            '''保存权重的文件夹'''
            folder_name = os.path.join("/home/user/SUN/HOAEXPERIMENT/Weightsave0/Ours/",
                                       f"seed={seed0}_index={index01 + 1}")
            os.makedirs(folder_name, exist_ok=True)
            # 以准确率命名模型权重文件名
            weight_filename = f"{SKF - 1}_acc={best_acc:.4f}.pt"
            weight_path = os.path.join(folder_name, weight_filename)
            torch.save(best_net.state_dict(), weight_path)
            '''中间结果'''
            Epoch = Epoch.values()
            Epoch = list(Epoch)
            Epoch = np.array(Epoch)
            acc_val_local = acc_val_local.values()
            acc_val_local = list(acc_val_local)
            acc_val_local = np.array(acc_val_local)
            Epoch_each = np.reshape(Epoch, (1, num_epoch))
            acc_val_each_local = np.reshape(acc_val_local, (1, num_epoch))
            print('局部分类器结果', acc_val_each_local)
            acc_val_global = acc_val_global.values()
            acc_val_global = list(acc_val_global)
            acc_val_global = np.array(acc_val_global)

            acc_val_each_global = np.reshape(acc_val_global, (1, num_epoch))
            print('全局分类器结果', acc_val_each_global)
        print(acc_per)
        print("十折的平均准确率为：",sum1/5)


