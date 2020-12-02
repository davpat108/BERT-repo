from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import pandas as pd



class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h = self.input_layer(X)
        h = self.relu(h)
        out = self.output_layer(h)
        return out



class Parameters():


    def __init__(self, epoch, Bertlayernumber):
        self.numepoch=epoch
        if Bertlayernumber>13:
            self.Bertlayernumber = 13
        elif Bertlayernumber<0:
            self.Bertlayernumber = 0
        else:
            self.Bertlayernumber=Bertlayernumber
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.BertModel = AutoModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True, return_dict=True)
        self.BertModel.eval()
        self.Net = SimpleClassifier(
        input_dim=768,
        output_dim=6,
        hidden_dim=50
    )
        self.Net
        self.Net=self.Net.cuda()
        self.criterion = self.createLoss()
        self.optimizer = self.createOptimizer()
        self.scheduler = self.creatScheduler()



    def casetonumber(self, case):
        switch = {
            "Nom": 0,
            "Acc": 1,
            "Ins": 2,
            "Ine": 3,
            "Sup": 4,
            "Sub": 5
        }
        return switch.get(case, "Invalid case")

    def gettokennumber(self, sentence, wordnumber):

        wordlist = sentence.split(" ")
        tokennumber = 0

        for i, word in enumerate(wordlist, 0):
            output = self.tokenizer(word, add_special_tokens=False)
            tokennumber = tokennumber + len(output["input_ids"])
            if i == int(wordnumber):
                break
        return tokennumber

    def preparedata(self):

        train_tsv = pd.read_csv('train.tsv', na_filter=None, quoting=3, sep="\t")
        dev_tsv = pd.read_csv('dev.tsv', na_filter=None, quoting=3, sep="\t")
        training_data = []
        test_data = []

        for i, obj in enumerate(train_tsv.values, 0):
            input = self.tokenizer(obj[0], padding=True,
                              truncation=True, return_tensors="pt")
            output = self.BertModel(**input)
            training_data.append([])

            training_data[i].append(output.hidden_states[self.Bertlayernumber][0][self.gettokennumber(obj[0], obj[2])].detach())
            training_data[i].append(self.casetonumber(obj[3]))

        for i, obj in enumerate(dev_tsv.values):
            input = self.tokenizer(obj[0], padding=True,
                              truncation=True, return_tensors="pt")
            output = self.BertModel(**input)
            test_data.append([])
            test_data[i].append(output.hidden_states[self.Bertlayernumber][0][self.gettokennumber(obj[0], obj[2])].detach())
            test_data[i].append(self.casetonumber(obj[3]))

        self.trainloader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True)

    def createLoss(self):
        return nn.CrossEntropyLoss()

    def createOptimizer(self):
        return torch.optim.SGD(
            self.Net.parameters(), lr=1e-1,
            momentum=0.9, nesterov=True,
            weight_decay=1e-4)

    def creatScheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.numepoch, eta_min=1e-2)

    def train(self, epoch):
        self.Net.train()
        running_loss = 0.0
        correct = 0.0
        total = 0

        for i, data in enumerate(self.trainloader, 0):
            tensors, labels = data

            tensors = tensors.cuda()
            labels = labels.cuda()

            self.optimizer.zero_grad()
            outputs = self.Net(tensors)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        tr_loss = running_loss / i
        tr_corr = correct / total * 100
        print("Train epoch " + str(epoch + 1) + "  correct: " + str(tr_corr))
        return tr_loss, tr_corr

    def val(self, epoch):
        self.Net.eval()
        running_loss = 0.0
        correct = 0.0
        total = 0

        for i, data in enumerate(self.testloader, 0):
            tensors, labels = data

            tensors = tensors.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                outputs = self.Net(tensors)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / i
        val_corr = correct / total * 100
        print("Test epoch " + str(epoch + 1) + " loss: " + str(running_loss / i) + " correct: " + str(val_corr))
        return val_loss, val_corr



