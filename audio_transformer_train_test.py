import time
import os
import numpy as np
import librosa
from pathlib import Path
import pandas as pd
import torchaudio
from torchvision.transforms import Compose
import zipfile
from torchaudio.transforms import Resample
from torchmetrics import Accuracy, ConfusionMatrix, AUROC, F1Score
import IPython.display as ipd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import math
import gdown

#%%

class CustomDataset(Dataset):
    def __init__(self, audio_files, sample_duration=2, overlap=0.5, label_mapping=None, transform=None):
        self.audio_files = audio_files
        self.sample_duration = sample_duration
        self.overlap = overlap
        self.transform = transform
        self.label_mapping=label_mapping

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        sample, sample_rate = torchaudio.load(self.audio_files[idx][0])
        if self.label_mapping:
            target = self.label_mapping[self.audio_files[idx][1]]
        else:
            target = self.audio_files[idx][1]
        
        if self.transform:
            sample = self.transform(torch.Tensor(sample))

        return sample,target


def collate_fn(batch):
    samples = [item[0] for item in batch]
    targets = torch.Tensor([item[1] for item in batch]).to(torch.long)

    stacked_samples = torch.stack([sample for sample in samples])
    
    return stacked_samples, targets

class AudioLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3,num_classes=10,loss_func=nn.CrossEntropyLoss):
        super(AudioLightning, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.criterion = loss_func()
        self.accuracy = Accuracy(task='multiclass',num_classes=self.num_classes)
        self.confusion_matrix = ConfusionMatrix(task="multiclass",num_classes=self.num_classes)
        self.auroc = AUROC(task="multiclass",num_classes=self.num_classes)
        self.f1 = F1Score(task="multiclass",num_classes=self.num_classes,average="weighted")
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.val_f1scores = []
        self.test_f1scores = []
        self.val_preds=[]
        self.test_preds=[]
        self.val_labels=[]
        self.test_labels=[]
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        wandb.log({'train_loss': loss},commit=True)
        acc = self.accuracy(y_hat, y)
        wandb.log({'val_accuracy':acc},commit=True)
        self.train_accuracies.append(acc.numpy())
        self.train_losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        pred = torch.argmax(y_hat,dim=1)
        wandb.log({'val_loss': loss},commit=True)
        acc = self.accuracy(y_hat, y)
        wandb.log({'val_accuracy': acc.numpy()},commit=True)
        f1=(self.f1(y_hat,y))
        wandb.log({'val_f1':f1.numpy()},commit=True)
        self.val_accuracies.append(acc.numpy())
        self.val_losses.append(loss.detach())
        self.val_f1scores.append((f1))

        self.val_labels.extend(list(y))
        self.val_preds.extend(list(y_hat))
        return {'val_labels':y,'val_preds':y_hat,'val_loss': loss, 'val_accuracy': acc.numpy(), 'val_f1':f1}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        wandb.log({'test_loss': loss},commit=True)
        acc=self.accuracy(y_hat, y)
        wandb.log({'test_accuracy': acc.numpy()},commit=True)
        f1=(self.f1(y_hat,y))
        wandb.log({'test_f1':f1.numpy()},commit=True)
        self.test_accuracies.append(acc.numpy())
        self.test_losses.append(loss.detach())
        self.test_f1scores.append(f1)
        self.test_labels.extend(list(y))
        self.test_preds.extend(list(y_hat))
        return {'test_labels':y,'test_preds': y_hat,'test_loss': loss, 'test_accuracy': acc.numpy(),'test_f1':f1}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_validation_end(self):
        wandb.log({'val_auc': wandb.plot.roc_curve(torch.Tensor(self.val_labels).numpy().astype(int),torch.stack(self.val_preds).numpy())})
        wandb.log({'val_cm': wandb.plot.confusion_matrix(y_true=torch.Tensor(self.val_labels).numpy().astype(int),preds=torch.argmax(torch.stack(self.val_preds),dim=1).numpy() )})
        return
    
    def on_test_end(self):
        wandb.log({'test_auc': wandb.plot.roc_curve(torch.Tensor(self.test_labels).numpy().astype(int),torch.stack(self.test_preds).numpy())})
        wandb.log({'test_cm': wandb.plot.confusion_matrix(y_true=torch.Tensor(self.test_labels).numpy().astype(int),preds= torch.argmax(torch.stack(self.test_preds),dim=1).numpy() )})
        return

class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)
    
class CNNBase(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CNNBase,self).__init__()

        self.layer1 =  nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=10,stride=5),
                nn.Dropout(0.1),
                Transpose(),
                nn.LayerNorm(out_channels),
                Transpose(),
                nn.GELU()
        )

        self.layer2 =  nn.Sequential(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=2),
                nn.Dropout(0.1),
                Transpose(),
                nn.LayerNorm(out_channels),
                Transpose(),
                nn.GELU()
        )

        self.layer3 =  nn.Sequential(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=2),
                nn.Dropout(0.1),
                Transpose(),
                nn.LayerNorm(out_channels),
                Transpose(),
                nn.GELU()
        )

        self.layer4 =  nn.Sequential(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=2),
                nn.Dropout(0.1),
                Transpose(),
                nn.LayerNorm(out_channels),
                Transpose(),
                nn.GELU()
        )

        self.layer5 =  nn.Sequential(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=2),
                nn.Dropout(0.1),
                Transpose(),
                nn.LayerNorm(out_channels),
                Transpose(),
                nn.GELU()
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class PositionEncoding(nn.Module):
    def __init__(self, dim, kernel_size, p_drop=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pos_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.GELU()
        )
        self.dropout = nn.Dropout(p_drop)
    
    def forward(self, x):
        x = x + self.pos_conv(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, p_drop=0.):
        super().__init__()
        self.heads = heads

        self.to_keys = nn.Linear(dim, dim)
        self.to_queries = nn.Linear(dim, dim)
        self.to_values = nn.Linear(dim, dim)
        self.unifyheads = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(p_drop)
        self.resid_drop = nn.Dropout(p_drop)

    def forward(self, x):
        b, t, d = x.size()
        h, d_q = self.heads, d // self.heads

        keys = self.to_keys(x).view(b, t, h, d_q).transpose(1, 2) 
        queries = self.to_queries(x).view(b, t, h, d_q).transpose(1, 2)
        values = self.to_values(x).view(b, t, h, d_q).transpose(1, 2)

        att = queries @ keys.transpose(-2, -1)
        att = nn.functional.softmax(att * d_q**-0.5, dim=-1)
        att = self.attn_drop(att)
        
        out = att @ values
        out = out.transpose(1, 2).contiguous().view(b, t, d) 
        out = self.unifyheads(out)
        out = self.resid_drop(out)
        return out


class Residual(nn.Module):
    def __init__(self, *layers):
        super(Residual,self).__init__()
        self.residual = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.residual(x)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, p_drop=0.):
        super(FeedForward,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p_drop)
        )
    def forward(self, x):
        return self.layer(x)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, p_drop):
        super(Block,self).__init__()

        self.stage1 = Residual(nn.LayerNorm(dim), MultiHeadSelfAttention(dim, heads, p_drop))
        self.stage2 = Residual(nn.LayerNorm(dim), FeedForward(dim, mlp_dim, p_drop))
        
    def forward(self,x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x

class TransformerBlocks(nn.Module):
    def __init__(self, dim, n_blocks, heads, mlp_dim, p_drop):
        super(TransformerBlocks,self).__init__()
        self.layers = [Block(dim, heads, mlp_dim, p_drop) for _ in range(n_blocks)]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim,n_blocks,n_heads):
        super(TransformerEncoder,self).__init__()
        self.posenc = PositionEncoding(dim, 65, 0.1)
        self.encoder = TransformerBlocks(dim, n_blocks, n_heads, 128, p_drop=0.1)
    def forward(self,x):
        x = self.posenc(x)
        x = self.encoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, dim, hidden_dim, num_classes, p_drop=0.):
        super(Classifier,self).__init__()
        self.norm = nn.Sequential(nn.LayerNorm(dim),
        Transpose(),
        nn.AdaptiveAvgPool1d(1))
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(p_drop))
        self.layer2 = nn.Linear(hidden_dim, num_classes)
    def forward(self,x):
        x = self.norm(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels,out_channels,dim,n_blocks,n_heads,num_classes,p_drop=0.1):
        super(Transformer,self).__init__()
        self.base = CNNBase(in_channels=in_channels,out_channels=out_channels)
        self.enc = TransformerEncoder(dim,n_blocks,n_heads)
        self.classifier = Classifier(dim, 128, num_classes, p_drop)
    
    def forward(self,x):
        x = self.base(x)
        x = self.enc(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params

# def predict(model, test_dataloader, device="cpu"):
#     model.to(device).eval()
#     predictions = []
#     labels = []
#     with torch.no_grad():
#         for batch,label in test_dataloader:
#             batch = batch.to(device)
#             outputs = model(batch)
#             probabilities = torch.softmax(outputs, dim=1)
#             _, predicted_class = torch.max(probabilities, dim=1)
#             predictions.extend(predicted_class.cpu().numpy())
#             labels.extend(label)
    
#     wandb.log({"conf_matrix": wandb.plot.confusion_matrix(y_true=labels,preds=predictions)})
#     return (predictions, labels)

if __name__ == "__main__":

    #Paramters
    input_dim = 1
    dim = 32
    n_transformer_blocks = 6
    pos_kernel_size = 65
    p_drop = 0.1

    test_samp = 1
    valid_samp = 2 
    batch_size = 40 
    num_workers = 2 
    learning_rate = 0.001
    num_classes = 10
    num_epochs = 5

    path = os.path.curdir
    df = pd.read_csv('meta/esc50.csv')
    df_esc10 = df[df["esc10"]==True][["filename", "target"]]
    
    wavs = ["audio/"+x for x in os.listdir(os.path.join(path,"audio"))]  
    esc10filenames = [os.path.basename(i) for i in df_esc10["filename"]]
    wavs = [(x,df_esc10.loc[df_esc10["filename"] == os.path.basename(x)]["target"].values[0]) for x in wavs if os.path.basename(x) in esc10filenames]

    unique_labels = np.unique([x[1] for x in wavs])
    label_mapping = {x:idx for idx,x in enumerate(unique_labels)}
    print(label_mapping)
    print(len(wavs))
    
    waveform, sample_rate = torchaudio.load(wavs[0][0])  
    n_channels = 64
    transform = torchaudio.transforms.Resample(44100,16000)
    
    dataset = CustomDataset(wavs,label_mapping=label_mapping,transform=transform)
    dataset_size = len(dataset)
    test_size = int((test_samp)/10 * dataset_size)
    train_size = len(dataset) - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    print(test_set.__getitem__(0)[1])

    k_folds = 4
    kfolder = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_datasets = []

    for train_index, val_index in kfolder.split(range(len(train_set))):
        fold_datasets.append((train_index, val_index))

    # HCTmodels_preds = []
    
    for fold_idx,fold in enumerate(fold_datasets):
        train_dataloader = DataLoader(Subset(train_set,fold[0]),batch_size=batch_size,shuffle=True, num_workers=num_workers,  persistent_workers=True,collate_fn=collate_fn)
        val_dataloader = DataLoader(Subset(train_set, fold[1]),batch_size=batch_size, num_workers=num_workers,  persistent_workers=True,collate_fn=collate_fn)
        test_dataloader = DataLoader(test_set,batch_size=batch_size, num_workers=num_workers,  persistent_workers=True,collate_fn=collate_fn)

        early_stop_callback = pl.callbacks.EarlyStopping(  
            monitor='val_loss',  
            min_delta=0.001,       
            patience=5,           
            verbose=True          
        )

        sample_x,label_x = next(iter(train_dataloader))
        n_channels = sample_x.shape[1]

        num_features=32
        audio_cnn_model = CNNBase(in_channels=n_channels,out_channels=num_features)
        x= audio_cnn_model(sample_x)

        d_model = x.shape[-1] * x.shape[-2]
        # wandb.init( project='ESC10_Audio_Classification',
        #             entity='m23csa010',
        #             config = {  "learning_rate":learning_rate,
        #                         "architecture":"CNN+FC",
        #                         "datasetFold":fold_idx,
        #                         "maxepochs":100,
        #                         })
        

        for n_heads in [4,2,1]:
            # HCTmodels_preds.append([])
            run=wandb.init( project='ESC10_Audio_Classification',
                        entity='m23csa010',
                        config = {  "learning_rate":learning_rate,
                                    "architecture":"CNN+Transformer+FC",
                                    "datasetFold":fold_idx,
                                    "maxepochs":num_epochs,
                                    "n_heads":n_heads
                                    })
            
            wandb_logger = pl.loggers.WandbLogger()
            
            print(f"Fold: {fold_idx}, Architecture: HCT, Num_Heads: {n_heads}")
            print("________________________________________________________________________")

            audio_hct_model = Transformer(n_channels,num_features,dim,n_blocks=n_transformer_blocks,num_classes=num_classes,n_heads=n_heads)
            lightning_model = AudioLightning(model=audio_hct_model, learning_rate=1e-3,loss_func=nn.CrossEntropyLoss)
            print(count_parameters(audio_hct_model))
            trainer = pl.Trainer(max_epochs=num_epochs, logger = wandb_logger)
            trainer.fit(lightning_model, train_dataloader, val_dataloader)
            trainer.test(lightning_model, test_dataloader)        

            run.finish()
            del audio_hct_model,lightning_model,trainer
        

    # labels = np.array(np.array([x[1] for x in HCTmodels_preds[0]]))
    # HCTcombinedpred = np.max(np.array([np.array(x[0]) for x in HCTmodels_preds]),axis=0)
    # # CNNcombinedpred = np.max(np.array([np.array(x[0]) for x in CNNmodels_preds]),axis=0)

    # print(labels)
    # print(HCTcombinedpred)
    # # print(CNNcombinedpred)

    # results_df = pd.DataFrame(zip(labels,HCTcombinedpred),columns=["y_true","y_hct"])
    # filenames = [os.path.basename(files) for files in os.listdir(os.path.curdir)]
    # if "predictions_run.csv" in filenames:
    #     iterator = 1
    #     while("predictions_run ("+str(iterator)+").csv" in filenames):
    #         iterator+=1
    #     results_df.save("predictions_run ("+str(iterator)+").csv")
    # print(accuracy_score(labels,HCTcombinedpred))
    # print(confusion_matrix(labels,HCTcombinedpred))
    # print(f1_score(labels,HCTcombinedpred,average="weighted"))
    # print(roc_auc_score(labels,HCTcombinedpred,average="weighted"))
    # print(accuracy_score(labels,CNNcombinedpred))


