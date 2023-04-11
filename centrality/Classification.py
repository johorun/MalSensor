from sqlite3 import Time
import networkx as nx
import time
import argparse
import csv
import numpy as np
#import tensorflow as tf
import os
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from itertools import islice
import warnings
warnings.filterwarnings("ignore")

knn1time = []
knn3time = []
rftime = []

def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains feature_CSV.', required=True)
    parser.add_argument('-o', '--output', help='The path of output.', required=True)
    parser.add_argument('-t', '--type', help='The type of centrality: degree, closeness, harmonic, katz', required=True)

    args = parser.parse_args()
    return args

def feature_extraction(file):
    vectors = []
    labels = []
    with open(file, 'r') as f:
        csv_data = csv.reader(f)    #把每行的元素作为列表元素返回
        for line in islice(csv_data, 1, None):      #从第二行开始读取数据
            vector = [float(i) for i in line[1:-1]] #每个程序的向量从第二位（第一位是sha256）开始到倒数第二位,每个元素作为列表的元素转换float
            label = int(float(line[-1]))            #最后一位是标签
            vectors.append(vector)                  #vectors=[[0,0,1,0,1,0...], [vector2], ... ]
            labels.append(label)

    return vectors, labels

def degree_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'degree_multi_features.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def katz_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'katz_multi_features.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def closeness_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'closeness_multi_features.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def harmonic_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'harmonic_multi_features.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def no_centrality_feature(feature_dir):
    feature_csv = feature_dir + 'no_multi_features.csv'
    vectors, labels = feature_extraction(feature_csv)
    return vectors, labels

def random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)): #软件个数
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec) #[[api1,api2,...,label], ...]

    random.shuffle(Vec_Lab) #随机排序

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]

from sklearn.neighbors import KNeighborsClassifier
def knn_1(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10) 
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    # TPRs = []
    # FPRs = []
    # TNRs = []
    # FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index] #train_X 是对应的训练向量组 train_Y 是标签，各有9/10个
        test_X, test_Y = X[test_index], Y[test_index] #对应的测试组，各有1/10个

        clf = KNeighborsClassifier(n_neighbors=1)
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        knn1time.append(tic1-tic)
        # tic = time.time()
        y_pred = clf.predict(test_X)
        # tic1 = time.time()
        # print((tic1-tic)/398)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        # print(f1)
        # TP = np.sum(np.multiply(test_Y, y_pred)) #对于0，1时求内积可以反映预测情况，但多分类时就不适用了
        # FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        # FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        # TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        # TPR = TP / (TP + FN)
        # FPR = FP / (FP + TN)
        # TNR = TN / (TN + FP)
        # FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        # TPRs.append(TPR)
        # FPRs.append(FPR)
        # TNRs.append(TNR)
        # FNRs.append(FNR)

    # print(F1s, FPRs)
    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

def knn_3(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    # TPRs = []
    # FPRs = []
    # TNRs = []
    # FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=3)
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        knn3time.append(tic1-tic)
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        # print(f1)
        # TP = np.sum(np.multiply(test_Y, y_pred))
        # FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        # FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        # TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        # TPR = TP / (TP + FN)
        # FPR = FP / (FP + TN)
        # TNR = TN / (TN + FP)
        # FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        # TPRs.append(TPR)
        # FPRs.append(FPR)
        # TNRs.append(TNR)
        # FNRs.append(FNR)

    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

from sklearn.ensemble import RandomForestClassifier
def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    # TPRs = []
    # FPRs = []
    # TNRs = []
    # FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=64, random_state=0)
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        rftime.append(tic1-tic)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        # print(f1)
        # TP = np.sum(np.multiply(test_Y, y_pred))
        # FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        # FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        # TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        # TPR = TP / (TP + FN)
        # FPR = FP / (FP + TN)
        # TNR = TN / (TN + FP)
        # FNR = FN / (TP + FN)
        # print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        # TPRs.append(TPR)
        # FPRs.append(FPR)
        # TNRs.append(TNR)
        # FNRs.append(FNR)

    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

from sklearn.tree import DecisionTreeClassifier
def decisiontree(vectors, labels):   
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = DecisionTreeClassifier(max_depth=8, random_state=0)
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        rftime.append(tic1-tic)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)

    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB,CategoricalNB
def naivebayes(vectors,labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = BernoulliNB(force_alpha=True)
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        rftime.append(tic1-tic)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)

    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

from sklearn.svm import SVC,NuSVC,LinearSVC
def svm(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = SVC(C=600.0,degree=3,gamma='scale',kernel='rbf')
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        rftime.append(tic1-tic)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)

    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

from sklearn.neural_network import MLPClassifier
def mlp(vectors,labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]
        
        print(train_X.shape)
        #le = LabelEncoder()
        #train_Y = le.fit_transform(train_Y)

        clf = MLPClassifier(random_state=0, learning_rate_init=0.005)
        tic = time.time()
        clf.fit(train_X, train_Y)
        tic1 = time.time()
        print(tic1-tic)
        rftime.append(tic1-tic)

        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)

    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

def myMLP(vectors,labels):
    # 定义 MLP 模型类
    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # 创建 MLP 模型实例并指定输入维度和输出类别数
    model = MLP(input_dim=vectors, num_classes=labels)

    # 定义损失函数、优化器和评价指标
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    metrics = {'accuracy': nn.functional.softmax}

    # 定义训练参数
    num_epochs = 1000
    batch_size = 100
    val_ratio = 0.1

    # 将数据转换为 PyTorch 张量
    X_train_tensor = torch.Tensor(X_train)
    Y_train_tensor = torch.LongTensor(Y_train)
    X_val_tensor = torch.Tensor(X_val)
    Y_val_tensor = torch.LongTensor(Y_val)

    # 划分训练集和验证集
    dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_set, val_set = torch.utils.data.random_split(dataset, [int((1-val_ratio)*len(dataset)), int(val_ratio*len(dataset))])

    # 定义训练集和验证集的 DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 开始训练
    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_acc += metrics['accuracy'](outputs, targets)[0].item() * inputs.size(0)
        train_loss /= len(train_set)
        train_acc /= len(train_set)

        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        for inputs, targets in val_loader:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_acc += metrics['accuracy'](outputs, targets)[0].item() * inputs.size(0)
        val_loss /= len(val_set)
        val_acc /= len(val_set)

        # 打印训练和验证集上的损失和准确率
        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        





def classification(vectors, labels):
    Vectors, Labels = random_features(vectors, labels)

    csv_data = [[] for i in range(7)]
    csv_data[0] = ['ML_Algorithm', 'F1', 'Precision', 'Recall', 'Accuracy']
    """
    csv_data[1].append('KNN-1')
    csv_data[1].extend(knn_1(Vectors, Labels))
    csv_data[2].append('KNN-3')
    csv_data[2].extend(knn_3(Vectors, Labels))
    csv_data[3].append('Random Forest')
    csv_data[3].extend(randomforest(Vectors, Labels))
    csv_data[4].append('SVM')
    csv_data[4].extend(svm(Vectors, Labels))
    csv_data[5].append('Desicion Tree')
    csv_data[5].extend(decisiontree(Vectors, Labels))
    """
    csv_data[6].append('Multi-layer Perceptron')
    csv_data[6].extend(mlp(Vectors, Labels))

    # csv_data[8].append('dnn')
    # csv_data[8].extend(dnn(Vectors, Labels))

    return csv_data


def main():
    args = parseargs()
    feature_dir = args.dir
    out_put = args.output
    type = args.type

    if feature_dir[-1] == '/':
        feature_dir = feature_dir
    else:
        feature_dir += '/'

    if out_put[-1] == '/':
        out_put = out_put
    else:
        out_put += '/'

    if type == 'degree':
        degree_vectors, degree_labels = degree_centrality_feature(feature_dir)
        degree_csv_data = classification(degree_vectors, degree_labels)
        degree_reults = out_put + 'degree_result.csv'
        with open(degree_reults, 'w', newline='') as f_degree: # w
            csvfile = csv.writer(f_degree)
            csvfile.writerows(degree_csv_data)
    elif type == 'harmonic':
        harmonic_vectors, harmonic_labels = harmonic_centrality_feature(feature_dir)
        harmonic_csv_data = classification(harmonic_vectors, harmonic_labels)
        harmonic_reults = out_put + 'harmonic_result.csv'
        with open(harmonic_reults, 'w', newline='') as f_harmonic:
            csvfile = csv.writer(f_harmonic)
            csvfile.writerows(harmonic_csv_data)
    elif type == 'katz':
        katz_vectors, katz_labels = katz_centrality_feature(feature_dir)
        katz_csv_data = classification(katz_vectors, katz_labels)
        katz_reults = out_put + 'katz_result.csv'
        with open(katz_reults, 'w', newline='') as f_katz:
            csvfile = csv.writer(f_katz)
            csvfile.writerows(katz_csv_data)
    elif type == 'closeness':
        closeness_vectors, closeness_labels = closeness_centrality_feature(feature_dir)
        closeness_csv_data = classification(closeness_vectors, closeness_labels)
        closeness_reults = out_put + 'closeness_result.csv'
        with open(closeness_reults, 'w', newline='') as f_closeness:
            csvfile = csv.writer(f_closeness)
            csvfile.writerows(closeness_csv_data)
    elif type == 'no':
        no_vectors, no_labels = no_centrality_feature(feature_dir)
        no_csv_data = classification(no_vectors, no_labels)
        no_reults = out_put + 'no_result.csv'
        with open(no_reults, 'w', newline='') as f_no:
            csvfile = csv.writer(f_no)
            csvfile.writerows(no_csv_data)
    else:
        print('Error Centrality Type!')


if __name__ == '__main__':
    main()

knn1total = 0
knn3total = 0
rftotal = 0
for ele in knn1time:
    knn1total = knn1total + ele
for ele in knn3time:
    knn3total = knn3total + ele
for ele in rftime:
    rftotal = rftotal + ele

print(knn1total, knn3total, rftotal)