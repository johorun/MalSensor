from sqlite3 import Time
import networkx as nx
import time
import argparse
import csv
import numpy as np
import os
from multiprocessing import Pool as ThreadPool
from functools import partial
import glob
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from itertools import islice
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
        csv_data = csv.reader(f)   
        for line in islice(csv_data, 1, None):      
            vector = [float(i) for i in line[1:-1]] 
            label = int(float(line[-1]))            
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

    for i in range(len(vectors)): 
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec) #[[api1,api2,...,label], ...]

    random.shuffle(Vec_Lab) 

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
        train_X, train_Y = X[train_index], Y[train_index] 
        test_X, test_Y = X[test_index], Y[test_index] 

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
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='micro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='micro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='micro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred, )


        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)


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
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='micro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='micro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='micro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        
        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)


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
        f1 = f1_score(y_true=test_Y, y_pred=y_pred, average='micro')
        precision = precision_score(y_true=test_Y, y_pred=y_pred, average='micro')
        recall = recall_score(y_true=test_Y, y_pred=y_pred, average='micro')
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)



        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)


    print(F1s)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys)]

def classification(vectors, labels):
    Vectors, Labels = random_features(vectors, labels)

    csv_data = [[] for i in range(4)]
    csv_data[0] = ['ML_Algorithm', 'F1', 'Precision', 'Recall', 'Accuracy']
    csv_data[1].append('KNN-1')
    csv_data[1].extend(knn_1(Vectors, Labels))
    csv_data[2].append('KNN-3')
    csv_data[2].extend(knn_3(Vectors, Labels))
    csv_data[3].append('Random Forest')
    csv_data[3].extend(randomforest(Vectors, Labels))

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
        with open(degree_reults, 'w', newline='') as f_degree:
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
