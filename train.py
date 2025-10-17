import torch 
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms,models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""训练二分类器(仅包含CHC和白细胞数据)，置信度低的归为癌细胞的train.py (using resnet18)"""

# TRAIN_FILE_PATHS = [
#     '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_1.csv',
#     '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_2.csv',
#     '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_3.csv'
# ]
TRAIN_FILE_PATHS = [
        '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_1.csv'
    ]
VAL_FILE_PATH = '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_4.csv'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)
        ])
    
    return transform

class cellDataset(Dataset):
    """自定义数据集类（使用均值填充）、自定义标签分配"""
    def __init__(self, dataframe, transform=None, target_size=224, num_classes=3):
        
        self.data = dataframe
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.data)
    
    def pad_image_with_mean(self, image, target_size):
        """使用均值填充将图像调整为目标大小"""
        h, w = image.shape[:2]
        
        if h >= target_size and w >= target_size:
            return cv2.resize(image, (target_size, target_size))
        # 每个通道的平均颜色
        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)
        padded_image = np.ones((target_size, target_size, 3), dtype=np.uint8)
        padded_image[:] = mean_color
        
        scale = min(target_size / h, target_size / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 等比缩放后的图像
        resized_image = cv2.resize(image, (new_w, new_h))
        
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 将等比缩放后的图像放在平均颜色的正方形画布上
        padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
        
        return padded_image
    
    def __getitem__(self,idx):
        row = self.data.iloc(idx)
        img_path = row["image_file_path"]
        cell_type = row["cell_type"]
        
        # CHC -> 1, 白细胞 -> 0
        label = 0 if cell_type!="CHC" else 1
        
        original_img = cv2.imread(img_path)
        padded_img = self.pad_image_with_mean(original_img, self.target_size)
        img = Image.fromarray(padded_img)
        
        if self.transform:
            img = self.transform(img)
        return img,label
    
    


def load_csv(train_csv_paths, val_csv_path):
    """ 返回训练集和验证集的df"""
    
    # 加载训练集和验证集
    print('\nloading training dataset...')
    train_dfs = []
    for train_path in train_csv_paths:
        df = pd.read_csv(train_path)
        train_dfs.append(df)
        print(f"{train_path}:{len(df)}个训练样本")
    train_df = pd.concat(train_dfs,ignore_index=True)
    
    print("loading validation dataset...")
    val_df = pd.read_csv(val_csv_path)
    print(f"{val_csv_path}:{len(val_df)}个验证样本")
    
    # 只保留需要的数据列
    required_cols = ['image_file_path', 'cell_type', 'x', 'y', 'w', 'h']
    train_df = train_df[required_cols]
    val_df = val_df[required_cols]
    
    # 打印统计信息
    print("\n"+"="*60)
    print(f"总训练样本数:{len(train_df)}")
    print("\n细胞类别分布:")
    print(train_df['cell_type'].value_counts())
    
    print("\nbinary classification (excluding CTC)...")
    train_df = train_df[train_df['cell_type'].isin(['CD66b','CD14','WBC','CD3','CHC'])]
    val_df = val_df[val_df['cell_type'].isin(['CD66b','CD14','WBC','CD3','CHC'])]
    
    print('total count of training data (CTC excluded): ',len(train_df))
    
    train_0 = train_df[train_df['cell_type']=='CHC']
    train_1 = train_df[train_df['cell_type']!='CHC']
    
    print(f"count of CHC: {len(train_0)}")
    print(f'count of WBC (including other type of wbc): {len(train_1)}')
    
    print("\n"+"="*60)
    
    # 展示一张图片
    # showImg(train_df)
    return train_df, val_df


def showImg(df):
    row = df.iloc[0]
    img_path = row['image_file_path']
    img = Image.open(img_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("./preview.png")
    
  

def train_epoch(model,dataloader,criterion,optimizer,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (img, labels) in enumerate(dataloader):
        img = img.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()   #把一个元素的张量转成python标量
        
        # output.shape == [B, 2]
        _, predicted = output.max(1) # 返回第二维度的最大值(置信度高的那一类),和它所属的类别predicted
        
        total += len(labels)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx+1)%50 ==0:
            print({f"Batch [{batch_idx+1}/{len(dataloader)}], Loss = {running_loss:.4f}, Acc = {100.* correct/total:.2f}%"})
   
    epoch_loss = running_loss/len(dataloader)
    epoch_acc = 100.*correct/total
    
    return epoch_loss, epoch_acc

# 验证函数
# def validate()

def main():
    lr = 1e-3
    batch_size = 64
    epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_size = 224
    confident_threshold = 0.7
    
    print("="*60)
    # 1. 加载数据
    train_df, val_df = load_csv(train_csv_paths=TRAIN_FILE_PATHS, val_csv_path=VAL_FILE_PATH)
    train_dataset_binary = cellDataset(train_df,transform=get_transforms(train=True),target_size=224,num_classes=2)
    val_dataset_binary = cellDataset(val_df, transform=get_transforms(train=False), target_size=target_size, num_classes=2)
    
    train_loader_binary = DataLoader(train_dataset_binary, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_binary = DataLoader(val_dataset_binary, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建二分类类型
    model_binary = models.resnet18(pretrained=True)
    num_features = model_binary.fc.in_features
    
    model_binary.fc = nn.Linear(num_features, 2)
    model_binary.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer_binary = optim.Adam(model_binary.parameters(),lr=lr)
    scheduler_binary = optim.lr_scheduler.ReduceLROnPlateau(optimizer_binary, mode='min', 
                                                             patience=5, factor=0.5, verbose=True)
    print("start training (binary model)...")
    
    
    
    
    

    
          

# load_csv(train_csv_paths=TRAIN_FILE_PATHS, val_csv_path=VAL_FILE_PATH)
if __name__=='__main__':
    main()
    
    

    
    
    
        


        
    
    
        
