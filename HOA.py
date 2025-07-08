import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def Mydataset():
    file1 = '/home/user/SUN/usingHOA/01HOA_LHipAngles_X.arff'
    file2 = '/home/user/SUN/usingHOA/02HOA_LHipAngles_Y.arff'
    file3 = '/home/user/SUN/usingHOA/03HOA_LHipAngles_Z.arff'
    file4 = '/home/user/SUN/usingHOA/41HOA_M0_LKneeAngles_X.arff'
    file5 = '/home/user/SUN/usingHOA/42HOA_M0_LKneeAngles_Y.arff'
    file6 = '/home/user/SUN/usingHOA/43HOA_M0_LKneeAngles_Z.arff'
    file7 = '/home/user/SUN/usingHOA/21HOA_M0_LAnkleAngles_X.arff'
    file8 = '/home/user/SUN/usingHOA/22HOA_M0_LAnkleAngles_Y.arff'
    file9 = '/home/user/SUN/usingHOA/23HOA_M0_LAnkleAngles_Z.arff'
    file10 = '/home/user/SUN/usingHOA/11HOA_M0_RHipAngles_X.arff'
    file11 = '/home/user/SUN/usingHOA/12HOA_M0_RHipAngles_Y.arff'
    file12 = '/home/user/SUN/usingHOA/13HOA_M0_RHipAngles_Z.arff'
    file13 = '/home/user/SUN/usingHOA/51HOA_M0_RKneeAngles_X.arff'
    file14 = '/home/user/SUN/usingHOA/52HOA_M0_RKneeAngles_Y.arff'
    file15 = '/home/user/SUN/usingHOA/53HOA_M0_RKneeAngles_Z.arff'
    file16 = '/home/user/SUN/usingHOA/31HOA_M0_RAnkleAngles_X.arff'
    file17 = '/home/user/SUN/usingHOA/32HOA_M0_RAnkleAngles_Y.arff'
    file18 = '/home/user/SUN/usingHOA/33HOA_M0_RAnkleAngles_Z.arff'

    file19 = '/home/user/SUN/usingHOA/HEA_M0_LHipAngles_X.arff'
    file20 = '/home/user/SUN/usingHOA/HEA_M0_LHipAngles_Y.arff'
    file21 = '/home/user/SUN/usingHOA/HEA_M0_LHipAngles_Z.arff'
    file22 = '/home/user/SUN/usingHOA/HEA_M0_LKneeAngles_X.arff'
    file23 = '/home/user/SUN/usingHOA/HEA_M0_LKneeAngles_Y.arff'
    file24 = '/home/user/SUN/usingHOA/HEA_M0_LKneeAngles_Z.arff'
    file25 = '/home/user/SUN/usingHOA/HEA_M0_LAnkleAngles_X.arff'
    file26 = '/home/user/SUN/usingHOA/HEA_M0_LAnkleAngles_Y.arff'
    file27 = '/home/user/SUN/usingHOA/HEA_M0_LAnkleAngles_Z.arff'
    file28  = '/home/user/SUN/usingHOA/HEA_M0_RHipAngles_X.arff'
    file29 = '/home/user/SUN/usingHOA/HEA_M0_RHipAngles_Y.arff'
    file30 = '/home/user/SUN/usingHOA/HEA_M0_RHipAngles_Z.arff'
    file31 = '/home/user/SUN/usingHOA/HEA_M0_RKneeAngles_X.arff'
    file32 = '/home/user/SUN/usingHOA/HEA_M0_RKneeAngles_Y.arff'
    file33 = '/home/user/SUN/usingHOA/HEA_M0_RKneeAngles_Z.arff'
    file34 = '/home/user/SUN/usingHOA/HEA_M0_RAnkleAngles_X.arff'
    file35 = '/home/user/SUN/usingHOA/HEA_M0_RAnkleAngles_Y.arff'
    file36 = '/home/user/SUN/usingHOA/HEA_M0_RAnkleAngles_Z.arff'



    dict1 = {
        '1': ['HOA2', 'HOA9', 'HOA11', 'HOA40', 'HOA42', 'HOA48',
              'HOA53', 'HOA55', 'HOA60', 'HOA70', 'HOA79', 'HOA83', 'HOA85',
              'HOA93', 'HOA103', 'HOA107', 'HOA112', 'HOA117'],
        '3': ['HOA1', 'HOA3', 'HOA5', 'HOA6', 'HOA7', 'HOA17',
              'HOA19', 'HOA20', 'HOA22', 'HOA24', 'HOA25', 'HOA26', 'HOA34'
            , 'HOA35', 'HOA39', 'HOA44', 'HOA45', 'HOA47', 'HOA52', 'HOA54', 'HOA57', 'HOA59'
            , 'HOA61', 'HOA63', 'HOA71', 'HOA78', 'HOA87', 'HOA88', 'HOA89', 'HOA91', 'HOA696'
            , 'HOA97', 'HOA98', 'HOA100', 'HOA104', 'HOA109', 'HOA116']}
    all_data01 = []
    all_data02 = []
    # 使用 for 循环依次处理 file1 到 file16
    for i in range(1, 19):
        file_name = locals()[f'file{i}']  # 构造文件名
        a1 = read_arff(file_name)  # 调用处理函数
        data01 = read(a1)
        # print(data01.shape)
        all_data01.append(data01)
    X1 = np.hstack(all_data01)  # 或者 np.concatenate(all_data, axis=1)
    # print(X1.shape)
    for i in range(19, 37):
        file_name = locals()[f'file{i}']  # 构造文件名
        a1 = read_arff(file_name)  # 调用处理函数
        data01 = read02(a1)
        # print(data01.shape)
        all_data02.append(data01)
    X2 = np.hstack(all_data02)  # 或者 np.concatenate(all_data, axis=1)
    # print(X2.shape)
    X = np.concatenate((X1, X2), axis=0)
    has_nan = np.isnan(X).any()

    # 输出是否存在 NaN
    if has_nan:
        print("数组中存在 NaN 值")
    else:
        print("数组中没有 NaN 值")
    nan_indices = np.argwhere(np.isnan(X))
    print("NaN 值的位置：", nan_indices)

    X = X.reshape(-1, 18, 101)#划分每份为101长度，总共18份

    print('样本总数与数据长度:', X.shape)  # 结果的形状应该是 (1050, 100 * 16)
    missing_hoas = [
        "HOA8", "HOA10", "HOA16", "HOA18", "HOA28", "HOA32", "HOA33", "HOA49", "HOA36", "HOA37",
        "HOA51", "HOA62", "HOA65", "HOA68", "HOA69", "HOA95", "HOA99", "HOA105", "HOA113"]

    hoa_names = [f"HOA{(i)}" for i in range(1, 121) if
                 f"HOA{(i)}" not in missing_hoas]  # 假设有 1050 个样本，对应 HOA1, HOA2, ..., HOA18
    # 定义缺失的 HOA 名称
    hoa_names.append("HOA201")

    # hoa_names = [hoa for hoa in hoa_names if hoa not in missing_hoas]

    # 输出结果
    # print(hoa_names)

    # 创建标签数组
    y = []
    # 根据每个样本的 HOA 名称为其分配标签
    for hoa in hoa_names:
        label = None
        for key, values in dict1.items():
            # print(values)
            if hoa in values:
                label = int(key)
                break
        if label is None:
            label = 2
        y.append(label)
    y.extend([0] * 80)
    y = np.array(y)
    Y = np.repeat(y, 10)
    # Y = [int(x) for x in Y]

    print('标签:', Y)
    print('标签维度：', Y.shape)
    has_nan = np.isnan(Y).any()

    # 输出是否存在 NaN
    if has_nan:
        print("数组中存在 NaN 值")
    else:
        print("数组中没有 NaN 值")
    nan_indices = np.argwhere(np.isnan(Y))
    print("NaN 值的位置：", nan_indices)
    # 创建一个包含1到284的数组
    groups = np.arange(1, 183).repeat(10)
    # # 将前102个数字各自重复20次
    # first_part = np.repeat(groups[:102], 20)
    # # 将后82个数字各自重复10次
    # second_part = np.repeat(groups[102:], 10)
    # # 合并两个部分
    # groups = np.concatenate((first_part, second_part))

    print(groups.shape)
    return X,Y,groups

def read_arff(file):
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith(("@attribute", "@ATTRIBUTE")):
                header.append(line.split()[1])
            elif line.startswith(("@data", "@DATA")):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df
def read(a):
    data_x_o = np.empty(shape=(0, 101))  # 堆叠数据
    for hoa in range(1, 202):
        if hoa in [36,37,33,68]:
            continue
        hoa_key = f"HOA{hoa}"
        hoa_data = a[a.iloc[:, -4] == hoa_key]  # Filter rows where the fourth column equals HOA{hoa}
        # Select only the first 10 rows (or fewer if there are less than 10 rows)
        # if 20>hoa_data.shape[0]>10:
        #     print(hoa_data.shape)
        #     print(hoa_key)
        hoa_data_10 = hoa_data.tail(10)
        data = np.array(hoa_data_10.iloc[:, 0:-4])
        # print(data)
        data_x_o = np.vstack([data_x_o, data])

    return data_x_o
def read02(a):
    data_x_o = np.empty(shape=(0, 101))  # 堆叠数据
    for hoa in range(121, 201):
        hoa_key = f"HEA{hoa}"
        hoa_data = a[a.iloc[:, -4] == hoa_key]  # Filter rows where the fourth column equals HOA{hoa}
        # Select only the first 10 rows (or fewer if there are less than 10 rows)
        hoa_data_10 = hoa_data.tail(10)
        data = np.array(hoa_data_10.iloc[:, 0:-4])
        # print(data.shape)
        data_x_o = np.vstack([data_x_o, data])

    return data_x_o


if __name__ == '__main__':


    X, Y, groups = Mydataset()
    # kf = StratifiedKFold(n_splits=7, shuffle=True)
    acc_fold = [0, 0, 0]
    best_score_fold = 0
    MATRIX_fold = 0
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True)
    sgkf.get_n_splits(X, Y)
    print(sgkf)
    # TODO:StratifiedGroupKFold(n_splits=3, random_state=None, shuffle=False)
    for i, (train_index, test_index) in enumerate(sgkf.split(X, Y, groups)):
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






