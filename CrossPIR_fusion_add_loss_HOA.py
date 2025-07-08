from math import ceil

from tody_layer import *
import numpy
import torch
from thop import profile
import torch.optim as optim

import time
def calculate_ait(model, input_tensor, runs=100):
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 预热 GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    ait = (total_time / runs) * 1000  # 转换成毫秒
    print(f"Average Inference Time (AIT): {ait:.3f} ms per batch")
    return ait
def calculate_att(model, input_tensor, target_tensor, runs=100):
    model.train()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 预热
    for _ in range(5):
        optimizer.zero_grad()
        output, _ ,_,_,_,_,_= model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(runs):
        optimizer.zero_grad()
        output, _ ,_,_,_,_,_ = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    att = (total_time / runs) * 1000  # 毫秒
    print(f"Average Training Time (ATT): {att:.3f} ms per batch")
    return att
class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Define layers for self attention mechanism
        self.query = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Calculate query, key, and value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Calculate attention scores
        attention_scores = F.softmax(torch.bmm(query, key), dim=-1)

        # Apply attention to value
        attention_output = torch.bmm(value, attention_scores.permute(0, 2, 1))
        attention_output = attention_output.view(batch_size, channels, height, width)

        # Apply gamma to attention output
        out = self.gamma * attention_output + x

        return out
class BasicResNetBlock1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size_conv1, padding_conv1):
        super(BasicResNetBlock1, self).__init__()
        # kernel_size_conv1 = kern_size
        # # kernel_size_conv2 = (5, 5)
        #
        # # Calculate 'same' padding
        # padding_conv1 = padding
        # padding_conv2 = ((kernel_size_conv2[0] - 1) // 2, (kernel_size_conv2[1] - 1) // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_conv1, padding=padding_conv1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_conv2, padding=padding_conv2)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # CBAM Attention
        self.cbam = CBAM(out_channels)

        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        # Apply CBAM attention
        out = self.cbam(out)

        # Shortcut connection
        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入 x 的维度为 (batch_size, in_channels, seq_len)
        batch_size, in_channels, seq_len = x.size()
        # 对输入进行全局平均池化，得到每个通道的特征表示
        y = self.avg_pool(x).view(batch_size, in_channels)
        # 使用全连接层学习每个通道的权重
        channel_weights = self.fc(y).view(batch_size, in_channels, 1)
        # 将学习到的权重应用到原始输入上，得到加权后的输出
        out = x * channel_weights
        return out
class BasicResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 1)):
        super(BasicResNetBlock, self).__init__()
        kernel_size_conv1 = (9, 9)
        kernel_size_conv2 = (5, 5)

        # Calculate 'same' padding
        padding_conv1 = ((kernel_size_conv1[0] - 1) // 2, (kernel_size_conv1[1] - 1) // 2)
        padding_conv2 = ((kernel_size_conv2[0] - 1) // 2, (kernel_size_conv2[1] - 1) // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_conv1, stride=strides, padding=padding_conv1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_conv2, padding=padding_conv2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # CBAM Attention
        self.cbam = CBAM(out_channels)

        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=strides, padding=0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply CBAM attention
        out = self.cbam(out)

        # Shortcut connection
        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)
        return out
class CBAM(nn.Module):
    def __init__(self, channels, reduction=10):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=(7, 7), padding=(3, 3), bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        avg_out = self.fc2(self.relu(self.fc1(avg_pool.view(avg_pool.size(0), -1))))
        max_out = self.fc2(self.relu(self.fc1(max_pool.view(max_pool.size(0), -1))))

        channel_attention = self.sigmoid_channel(avg_out + max_out).unsqueeze(2).unsqueeze(3)

        # Spatial Attention Module
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_conv = self.spatial_conv(spatial_concat)
        spatial_attention = self.sigmoid_spatial(spatial_conv)

        # Combine Channel and Spatial Attention
        attention = channel_attention * spatial_attention
        return x * attention

#融合部分加入CrossAttention

class GNNStack(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size,
                 in_dim, hidden_dim, out_dim,
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU()):

        super().__init__()
        #自行增加部分
        # self.bn0 = nn.BatchNorm2d(1)
        # self.conv1 = nn.Conv2d(1, 72, kernel_size=(1, 7), stride=(1, 2), padding=(0, 1))
        # self.bn1 = nn.BatchNorm2d(72)
        # self.relu = nn.ReLU(inplace=True)
        # self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

        # TODO: Sparsity Analysis稀疏性分析
        k_neighs = self.num_nodes = num_nodes

        self.num_graphs = groups

        self.num_feats = seq_len
        self.fc1 = nn.Linear(256, 128)
        if seq_len % groups:
            self.num_feats += (groups - seq_len % groups)
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)

        gnn_model, heads = self.build_gnn_model(gnn_model_type)

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [(k - 1) // 2 for k in kern_size]

        self.tconvs = nn.ModuleList(
            [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] +
            [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[layer + 1]), padding=(0, paddings[layer + 1])) for
             layer in range(num_layers - 2)] +
            [nn.Conv2d(heads * hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        )
        # self.tconvs = nn.ModuleList(
        #     [BasicResNetBlock1(72, in_dim, (1, kern_size[0]),(0, paddings[0]))] +
        #     [BasicResNetBlock1(heads * in_dim, hidden_dim, (1, kern_size[layer + 1]), (0, paddings[layer + 1])) for
        #      layer in range(num_layers - 2)] +
        #     [BasicResNetBlock1(heads * hidden_dim, out_dim, (1, kern_size[-1]), (0, paddings[-1]))]
        # )

        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] +
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 2)] +
            [gnn_model(out_dim, heads * out_dim, groups)]
        )

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(heads * in_dim)] +
            [nn.BatchNorm2d(heads * hidden_dim) for _ in range(num_layers - 2)] +
            [nn.BatchNorm2d(heads * out_dim)]
        )

        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round(num_nodes * (1 - (pool_ratio * layer)))
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        self.diffpool = nn.ModuleList(
            [Dense_TimeDiffPool2d(self.left_num_nodes[layer], self.left_num_nodes[layer + 1], kern_size[layer],
                                  paddings[layer]) for layer in range(num_layers - 1)] +
            [Dense_TimeDiffPool2d(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # self.linear = nn.Linear(heads * out_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            # tconv.reset_parameters()
            gconv.reset_parameters()
            bn.reset_parameters()
            pool.reset_parameters()

        # self.fc1.reset_parameters()

    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1

    def forward(self, inputs: Tensor):

        inputs1 = inputs.float()
        inputs1 = inputs1.transpose(1,2)
        # print(inputs1.shape)
        inputs1 = inputs1.unsqueeze(1)
        inputs1 = inputs1.transpose(2,3)
        # print(inputs1.shape)
                #resnet基线部分

        if inputs1.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs1.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs1, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs1

        adj = self.g_constr(x.device)
        # x02 = self.bn0(x)
        # x02 = self.conv1(x02)
        # x = self.relu(x02)


        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns,
                                          self.diffpool):  # 遍历图网络中使用的层或模块被zip函数一个个组合起来


            x, adj = pool(gconv(tconv(x), adj), adj)  # 分别进行时间卷积和图卷积，tconv先对x处理，gconv使用邻接矩阵adj对结果进行图卷积操作

            x = self.activation(bn(x))  # 标准化和激活函数操作

            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        # out = self.fc1(out)



        return out

class ResNet(nn.Module):
    def __init__(self, hidden_size, num_classes=3):
        super(ResNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(6)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Three ResNet blocks
        self.res_blocks = nn.Sequential(
            BasicResNetBlock(64, 128),
            BasicResNetBlock(128, 256),
            BasicResNetBlock(256, 512)
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.self_attention = SelfAttention(in_channels=512, reduction_ratio=8)
        self.channel_attention = ChannelAttention(in_channels=18)

        # Fully connected layers
        self.fc1 = nn.Linear(512, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = x.float()
        # x = x.transpose(1, 2)
        # x02 = self.channel_attention(x)
        # x = x+x02
        x = x.view(-1, 6, 3, 101)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # ResNet blocks
        x = self.bn1(x)
        x = self.res_blocks(x)
        # x = self.self_attention(x)

        # Global Average Pooling
        x = self.global_avg_pool(x).view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image1, image2):  # image1[2, 3, 224, 224], image2[2, 3, 224, 224]

        # 用transformer提取img1和img2的特征
        img_feat_1 = image1  # img_feat_1[2, 197, 768]
        img_patch_feat_1 = img_feat_1[:, :, :]  # img_patch_feat_1[2, 196, 768]
        img_feat_2 = image2  # img_feat_2[2, 197, 768]
        img_patch_feat_2 = img_feat_2[:, :, :]  # img_patch_feat_2[2, 196, 768]

        # 获取img1的带有交叉注意力的特征
        atten_sim_1 = torch.bmm(img_patch_feat_1, img_patch_feat_2.permute(0, 2, 1))  # atten_sim_1[2, 196, 196]
        atten_scores_1 = F.softmax(atten_sim_1, dim=-1)  # atten_scores_1[2, 196, 196]
        img_feat_2_atten_output = torch.bmm(atten_scores_1, img_patch_feat_2)  # img_feat_1_atten_output[2, 196, 768]
        img_feat_2_atten_output = F.normalize(img_feat_2_atten_output, dim=-1)+img_patch_feat_2  # img_feat_1_atten_output[2, 196, 768]  img1的最终输出
        # img_feat_2_atten_output = torch.cat([img_feat_2_atten_output, img_patch_feat_2], dim=1)
        # 获取img2的带有交叉注意力的特征
        atten_sim_2 = torch.bmm(img_patch_feat_2, img_patch_feat_1.permute(0, 2, 1))  # atten_sim_2[2, 196, 196]
        atten_scores_2 = F.softmax(atten_sim_2, dim=-1)  # atten_scores_2[2, 196, 196]
        img_feat_1_atten_output = torch.bmm(atten_scores_2, img_patch_feat_1)  # img_feat_2_atten_output[2, 196, 768]
        img_feat_1_atten_output = F.normalize(img_feat_1_atten_output, dim=-1)+img_patch_feat_1  # img_feat_2_atten_output[2, 196, 768]  img2的最终输出
        # img_feat_1_atten_output=torch.cat([img_feat_1_atten_output,img_patch_feat_1],dim=1)
        return img_feat_1_atten_output + image1, img_feat_2_atten_output+image2
class HeadLayer(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(HeadLayer, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

def similarity(feature1, feature2):
    feature1 = F.normalize(feature1, dim=-1)
    feature2 = F.normalize(feature2, dim=-1)
    bz = feature1.size(0)
    labels = torch.arange(bz).type_as(feature1).long()
    scores = feature1.mm(feature2.t())
    scores /= 0.07
    scores1 = scores.transpose(0, 1)#对分数进行转置，用于计算另一种方向的相似度分数
    loss0 = F.cross_entropy(scores, labels)
    loss1 = F.cross_entropy(scores1, labels)
    # loss0 = nn.CrossEntropyLoss(scores, labels)
    # loss1 = nn.CrossEntropyLoss(scores1, labels)

    loss_ita = (loss0 + loss1) / 2.0
    # loss_ita = loss1

    return loss_ita
class FusionModel(nn.Module):
    def __init__(self, hidden_size1, hidden_size2,  num_classes, num_dp):
        super(FusionModel, self).__init__()
        # Define two base models
        self.kern_size = [int(l) for l in "9,5,3".split(",")]
        self.base_model1 =GNNStack(gnn_model_type='dyGIN2d', num_layers=3,
                       groups=5, pool_ratio=0.2, kern_size=self.kern_size,
                       in_dim=64, hidden_dim=128, out_dim=256,
                       seq_len=101, num_nodes=18, num_classes=4)

        self.base_model2 = ResNet(hidden_size2)
        self.relu = nn.ReLU(inplace=False)

        # Define weights for feature fusion
        self.weight1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.weight2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        # Final fusion layer
        self.fusion_layer = nn.Linear(in_features=hidden_size1 + hidden_size2, out_features=num_classes)
        self.local_classifier = nn.Linear(in_features=hidden_size1,out_features=num_classes)
        self.global_classifier = nn.Linear(in_features=hidden_size2, out_features=num_classes)
        # self.optimizer = optim.SGD([self.weight1, self.weight2], lr=0.01)

        self.BN = nn.BatchNorm1d(hidden_size1 + hidden_size2)
        self.dp = nn.Dropout(num_dp)
        self.heads = nn.ModuleList([HeadLayer(2, 3) for _ in range(2)])
        self.combine = nn.Linear(512, num_classes)
        self.CrossAttention = CrossAttention()

    def reset_parameters(self):
        self.fusion_layer.reset_parameters()

    def forward(self, x):
        features_1 = self.base_model1(x)
        features_2 = self.base_model2(x)

        #新增的交叉注意力部分

        output_features1 = features_1.view(-1, 256, 1)
        output_features2 = features_2.view(-1, 256, 1)
        features1_0, features2_0 = self.CrossAttention(output_features1, output_features2)
        features1 = features1_0.view(features1_0.size()[0], -1)
        features2 = features2_0.view(features2_0.size()[0], -1)
        loss_ita = similarity(features1, features2)

        combined_output = torch.cat([features1_0,features2_0], dim=1)
        combined_output = combined_output.view(combined_output.size()[0], -1)

        output = self.fusion_layer(combined_output)
        output_local = self.local_classifier(features1)

        output_global = self.global_classifier(features2)

        output1 = nn.Softmax(dim=1)(output)

        return output_local, output_global, output, output1, features1, features2, loss_ita


if __name__ == "__main__":
    input_data = numpy.random.rand(1, 18, 101)
    y = torch.randint(0, 4, (1,))
    input_data = torch.Tensor(input_data)
    kern_size = [int(l) for l in "9,5,3".split(",")]
    net = FusionModel(hidden_size1=256, hidden_size2=256, num_classes=4, num_dp=0.3)
    output_local, output_global, output, output1, features1, features2, loss_ita =net(input_data)
    print(output_local.shape)
    flops, params = profile(net, inputs=(input_data,))  # 计算模型复杂度
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Calculate AIT
    calculate_ait(net, input_data, runs=100)

    # Calculate ATT
    calculate_att(net, input_data, y, runs=50)


