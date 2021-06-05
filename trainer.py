import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloader import ImgDataset
import pandas as pd


from newmodel import resnet18
import numpy as np
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda')
#device = torch.device('cpu')
#xx='1'
xx='official'
def save_dict(m):
    with open(xx+'best_model.pth', 'wb') as f:
        torch.save(m.state_dict(), f)


def load_dict(m,device):
    m.load_state_dict(torch.load(xx+'best_model.pth'))
    m.to(device)

def train(m,max_epochs,dataloader,o,s):

    best_average_wr = 0
    best_epoch = 0
    patience = 5

    for i in range(max_epochs):
        train_one_epoch(m,i,dataloader[0],o)
        average_wr = valid(m,dataloader[1])
        if average_wr>best_average_wr:
            best_average_wr=average_wr
            best_epoch = i
            save_dict(m)
        if i - best_epoch > patience:
            break
        s.step()


def train_one_epoch(m,epoch,dataloader,o):
    m.train()
    loss = torch.nn.CrossEntropyLoss()
    #loss = torch.nn.BCEWithLogitsLoss()
    epoch_loss=[]
    rate=[]
    for step, batch_data in enumerate(dataloader):
        o.zero_grad()
        img_tensor = batch_data[0].to(device)
        img_label = batch_data[1].to(device)
        img_label_index = batch_data[2].to(device)
        prediction = m(img_tensor)
        train_loss = loss(prediction,img_label_index.long())
        train_loss.backward()
        o.step()
        epoch_loss.append(train_loss.item())
        pred_class = torch.max(prediction, dim=1)[1]
        # print(pred_class)
        # print(img_label_index)
        img_label_index = batch_data[-1].to(device).view(-1)
        correct = 0
        for i in range(len(img_label_index)):
            # print(img_label_index[i],pred_class[i])
            # print(img_label_index[i]==pred_class[i])
            if img_label_index[i] == pred_class[i]:
                correct += 1
        rate.append(correct / len(pred_class))

        if (step % 30 == 0) and step > 0:

            print(f'Train epoch: {epoch}[{step}/{len(dataloader)}]'
                  f'({100. * step / len(dataloader):.0f}%)]\t ,'
                  f'Loss: {train_loss.item():.6f}, '
                  f'AvgL: {np.mean(epoch_loss):.6f},'
                  f'Acc:{correct / len(pred_class):.6f},'
                  f'AvgAcc:{np.mean(rate):.6f},')

def valid(m,dataloader):
    print('start valid')
    m.eval()
    rate = []
    for step, batch_data in enumerate(dataloader):
        img_tensor = batch_data[0].to(device)
        img_label = batch_data[1].to(device)
        img_label_index = batch_data[2].to(device).view(-1)
        prediction = m(img_tensor)
        pred_class = torch.max(prediction, dim=1)[1]
        #print(pred_class)
        #print(img_label_index)
        correct = 0
        for i in range(len(img_label_index)):
            #print(img_label_index[i],pred_class[i])
            #print(img_label_index[i]==pred_class[i])
            if img_label_index[i] == pred_class[i]:
                correct+=1
        rate.append(correct/len(pred_class))
    result = np.mean(rate)
    print('Valid:',result)
    return result

def test(m,dataloader,df):
    m.eval()
    res= []
    for step, batch_data in enumerate(dataloader):
        img_tensor = batch_data[0].to(device)
        img_label = batch_data[1].to(device)
        prediction = m(img_tensor)
        pred_class = torch.max(prediction, dim=1)[1].squeeze(-1)
        res.extend(pred_class.cpu().numpy().tolist())
    df['Category']=res
    df.to_csv('./'+xx+'result.csv',index=False)


def _initialize_weights(model):
    # print(self.modules())

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            # print(m.weight.data.type())
            # input()
            # m.weight.data.fill_(1.0)
            torch.nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    train_mode = True
    model = resnet18().to(device)
    #model = toy_model().to(device)
    #model = AlexNet(180).to(device)
    #model = Net().to(device)
    #_initialize_weights(model)

    torch.multiprocessing.set_sharing_strategy('file_system')
    if train_mode:
        train_df = pd.read_csv('./train.csv')
        valid_df = pd.read_csv('./valid.csv')
        test_df = pd.read_csv('./sampleSubmission.csv')
        train_data_loader = DataLoader(
            dataset=ImgDataset(train_df, './train',180),
            batch_size=70,
            shuffle=True,
            num_workers=64,
            pin_memory=True)
        valid_data_loader = DataLoader(
            dataset=ImgDataset(valid_df, './valid',180),
            batch_size=32,
            shuffle=False,
            num_workers=5,
            pin_memory=False)
        test_data_loader = DataLoader(
            dataset=ImgDataset(test_df, './test',180),
            batch_size=32,
            shuffle=False,
            num_workers=5,
            pin_memory=False)
        optimizer = torch.optim.SGD(model.parameters(),lr=0.05,momentum=0.4,weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer,gamma=0.9,step_size=5)

        loader=[train_data_loader,valid_data_loader,test_data_loader]
        train(model,2000,loader,optimizer,scheduler)
        load_dict(model,device)
        test(model,test_data_loader,test_df)
    else:
        test_df = pd.read_csv('./sampleSubmission.csv')
        test_data_loader = DataLoader(
            dataset=ImgDataset(test_df, './test',180),
            batch_size=32,
            shuffle=False,
            num_workers=5,
            pin_memory=False)
        load_dict(model,device)
        test(model,test_data_loader,test_df)