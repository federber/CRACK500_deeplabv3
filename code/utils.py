import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import math
import cv2
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

import torchmetrics as tm



def mirror_pad(image, target_size=(640, 640)):
    #приводит изображения к одному размеру путем центрирования и симметричного отражения краев
    height, width = image.shape[:2]
    target_width, target_height = target_size

    if width >= target_width or height >= target_height:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST if len(image.shape) == 2 else cv2.INTER_LINEAR)

    left_pad = max((target_width - width) // 2, 0)
    right_pad = max(target_width - width - left_pad, 0)
    top_pad = max((target_height - height) // 2, 0)
    bottom_pad = max(target_height - height - top_pad, 0)

    padded_image = cv2.copyMakeBorder(
        image, top_pad, bottom_pad, left_pad, right_pad,
        borderType=cv2.BORDER_REFLECT
    )

    return padded_image

class SegData(Dataset):
    def __init__(self, dataset_path, file_name, transform=None):
        self.images = []
        self.masks = []
        self.transform = transform
        self.dataset_path = dataset_path
        file_path = Path(dataset_path) / file_name
        # Читаем файлы из train.txt
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            image_path, mask_path = line.strip().split()
            self.images.append(image_path)
            self.masks.append(mask_path)

    def __getitem__(self, idx):
        image_name = Path(self.dataset_path) / self.images[idx]
        mask_name = Path(self.dataset_path) / self.masks[idx]
        image = cv2.imread(str(image_name))
        #print(image_name)
        #plt.imshow(torch.from_numpy(image))
        #plt.show()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_name), cv2.IMREAD_UNCHANGED)
        image = mirror_pad(image)
        #plt.imshow(torch.from_numpy(image))
        #plt.show()
        mask = mirror_pad(mask)

        #plt.imshow(torch.from_numpy(mask))
        #plt.show()
        image = image.astype('uint8')
        mask = mask.astype('uint8')
        mask = np.expand_dims(mask, axis=2)

        if(self.transform is not None):
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.float()/255
            mask = mask.squeeze().long()
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.float()/255
            mask = mask.squeeze().long()
        return image, mask
    def __len__(self):
        return len(self.images)
    

class DeepLabv3_trainer():
    def __init__(self,model,dataset_dir_path):
        self.model = model
        self.dataset_dir_path = dataset_dir_path
        
        train_ds = SegData(dataset_path=dataset_dir_path, file_name='train.txt', transform=None)
        val_ds = SegData(dataset_path=dataset_dir_path, file_name='val.txt', transform=None)
        test_ds = SegData(dataset_path=dataset_dir_path, file_name='test.txt', transform=None)

        self.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
        self.val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
        self.test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)



    def train_model(self, save_folder, epohs = 30, device = 'cuda'):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters())
        self.epochs = epohs
        self.device = device
        if self.device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = self.model.to(device)

        acc = tm.Accuracy(task="multiclass", average="micro", num_classes=2).to(device)
        f1 = tm.F1Score(task="multiclass", average="macro", num_classes=2).to(device)
        iou_metric = tm.JaccardIndex(task="binary").to(device)
        prec = tm.Precision(task="binary").to(device)
        rec = tm.Recall(task="binary").to(device)

        epohs_arr = []
        t_loss = []
        t_acc = []
        t_f1 = []
        t_prec = []
        t_rec = []
        v_loss = []
        v_acc = []
        v_f1 = []
        t_iou = []
        v_iou = []
        v_prec = []
        v_rec = []


        self.model.train()
        start_epoch = 0 #10

        start_f1VMax = 0 #0.8878
        f1VMax = 0.0 + start_f1VMax

        print('starts to learn')
        for epoch in range(start_epoch, start_epoch + self.epochs+1):
            running_loss = 0.0

            for inputs, targets in self.train_dl:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                acc.update(outputs, targets)
                f1.update(outputs, targets)
                iou_metric.update(torch.argmax(outputs, dim=1), targets)
                prec.update(torch.argmax(outputs, dim=1), targets)
                rec.update(torch.argmax(outputs, dim=1), targets)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()


            epoch_loss = running_loss / len(self.train_dl)
            accT = acc.compute()
            f1T = f1.compute()
            m_iouT = iou_metric.compute()
            precT = prec.compute()
            recT = rec.compute()
            

            print(f'Epoch: {epoch}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {accT:.4f}, Training F1: {f1T:.4f}, Training IoU: {m_iouT:.4f}')
            print(f'Training Precision: {precT:.4f}, Training Recall: {recT:.4f}')

            epohs_arr.append(epoch)
            t_loss.append(epoch_loss)
            t_acc.append(accT.detach().cpu().numpy())
            t_f1.append(f1T.detach().cpu().numpy())
            t_iou.append(m_iouT.detach().cpu().numpy())
            t_prec.append(precT.detach().cpu().numpy())
            t_rec.append(recT.detach().cpu().numpy())

            acc.reset()
            f1.reset()
            iou_metric.reset()
            prec.reset()
            rec.reset()

            with torch.no_grad():
                running_loss_v = 0.0

                for inputs, targets in self.val_dl:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = self.model(inputs)
                    loss_v = criterion(outputs, targets)

                    running_loss_v += loss_v.item()

                    acc.update(outputs, targets)
                    f1.update(outputs, targets)
                    iou_metric.update(torch.argmax(outputs, dim=1), targets)
                    prec.update(torch.argmax(outputs, dim=1), targets)
                    rec.update(torch.argmax(outputs, dim=1), targets)

            epoch_loss_v = running_loss_v/len(self.val_dl)
            accV = acc.compute()
            f1V = f1.compute()
            m_iouV = iou_metric.compute()
            precV = prec.compute()
            recV = rec.compute()

            print(f'Validation Loss: {epoch_loss_v:.4f}, Validation Accuracy: {accV:.4f}, Validation F1: {f1V:.4f}, Validation IoU: {m_iouV:.4f}')
            print(f'Validation Precision: {precV:.4f}, Validation Recall: {recV:.4f}')

            v_loss.append(epoch_loss_v)
            v_acc.append(accV.detach().cpu().numpy())
            v_f1.append(f1V.detach().cpu().numpy())
            v_iou.append(m_iouV.detach().cpu().numpy())
            v_prec.append(precV.detach().cpu().numpy())
            v_rec.append(recV.detach().cpu().numpy())


            acc.reset()
            f1.reset()
            iou_metric.reset()
            prec.reset()
            rec.reset()

            f1V2 = f1V.detach().cpu().numpy()
            if f1V2 > f1VMax:
                f1VMax = f1V2
                torch.save(self.model.state_dict(), Path(save_folder) / 'crack_dlv3_model.pt')
                print(f'Model saved for epoch {epoch}.')

        Sepohs = pd.Series(epohs_arr, name="epoch")
        St_loss = pd.Series(t_loss, name="training_loss")
        St_acc = pd.Series(t_acc, name="training_accuracy")
        St_f1 = pd.Series(t_f1, name="training_f1")
        St_iou = pd.Series(t_iou, name="training_iou")
        St_prec = pd.Series(t_prec, name="training_prec")
        St_rec = pd.Series(t_rec, name="training_rec")
        Sv_loss = pd.Series(v_loss, name="val_loss")
        Sv_acc = pd.Series(v_acc, name="val_accuracy")
        Sv_f1 = pd.Series(v_f1, name="val_f1")
        Sv_iou = pd.Series(v_iou, name="val_iou")
        Sv_prec = pd.Series(v_prec, name="val_prec")
        Sv_rec = pd.Series(v_rec, name="val_rec")


        resultsDF = pd.concat([Sepohs, St_loss, St_acc, St_f1, St_iou, St_prec, St_rec, Sv_loss, Sv_acc, Sv_f1, Sv_iou, Sv_prec, Sv_rec], axis=1)

        resultsDF.to_csv(Path(save_folder) / "results_crack.csv")