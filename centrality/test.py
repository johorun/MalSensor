import os
import random
import torch
import time
import json
import pickle
import numpy
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score


def set_seed(cuda_num = "0"):
    seed = 2022
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num

def sava_data(filename, data):
    print("saving data to:", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

import pickle
def load_data(filename):
    print("loading data from:", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def get_MCM_score(test_Y, y_pred):
    f_score = f1_score(y_true=test_Y, y_pred=y_pred, average='macro')
    precision = precision_score(y_true=test_Y, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=test_Y, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=test_Y, y_pred=y_pred, )

    return {
        "precision": format(precision * 100, '.3f'),
        "recall": format(recall * 100, '.3f'),
        "f_score" : format(f_score * 100, '.3f'),
        "ACC": format(acc * 100, '.3f')    }



class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.fc1 = nn.Linear(5246,256) #500
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TraditionalDataset(Dataset):
    def __init__(self, data_df):
        self.texts = data_df['vector']
        self.targets = data_df['label']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = int(self.targets[idx])-1
        return {
            'vector': torch.tensor(feature, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.long)
        }



class TextCNN_Classifier():
    def __init__(self, **kw):
        self.model = TextCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = 64
        self.learning_rate = 0.005
        self.epochs = 50
        

    def preparation(self, **kw):
        self.item_num = kw["item_num"]
        self.path_name = kw["path_name"] 
        res_path = os.path.join("/root/data/aigc/test/data/", self.path_name, self.path_name + "_10_fole")
        if not os.path.exists(res_path): os.makedirs(res_path)
        self.result_save_path = os.path.join(res_path, str(self.item_num) + ".result")
        # create datasets
        self.train_set = TraditionalDataset(kw["train_df"])
        self.valid_set = TraditionalDataset(kw["test_df"])

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
    
    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            targets = data["targets"].to(self.device)
            with autocast():
                outputs  = self.model( vectors )
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()           
            
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))   # 获取预测
            labels += list(np.array(targets.cpu()))      # 获取标签

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict
    
    
    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        with torch.no_grad():
            for i, data in progress_bar:
                vectors = data["vector"].to(self.device)
                targets = data["targets"].to(self.device)
                outputs = self.model(vectors)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    def train(self):
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'precision', 'recall', 'f_score', 'ACC', 'time'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'precision', 'recall', 'f_score', 'ACC', 'time'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            start = time.time()
            train_loss, train_score = self.fit()
            end = time.time()
            train_score["time"] = format(end-start, '.3f')
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "report_dict"])
            print(train_table)

            start = time.time()
            val_loss, val_score = self.eval()
            end = time.time()
            val_score["time"] = format(end-start, '.3f')
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "report_dict"])
            print(test_table)
            print("\n")

            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                    "train_score": train_score, "val_score": val_score}
            sava_data(self.result_save_path, learning_record_dict)
            print("\n")
        

def train_model(**kw):
    for path_name in ["degree_multi_features", "harmonic_multi_features", "katz_multi_features"]:
        for item_num in range(10):
            classifier = TextCNN_Classifier()
            train_df = load_data("/root/data/aigc/test/data/" + path_name +"/train.pkl")[item_num]
            test_df = load_data("/root/data/aigc/test/data/" + path_name + "/test.pkl")[item_num]
            classifier.preparation(train_df = train_df, test_df = test_df, item_num = item_num, path_name = path_name)
            classifier.train()
    


def main():
    #2e-5
    set_seed()
    train_model()
    print("hello")

if __name__ == "__main__":
    main()