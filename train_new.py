import torch 
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

"""训练二分类器(仅包含CHC和白细胞数据)，置信度低的归为癌细胞的train.py (using resnet18)"""

TRAIN_FILE_PATHS = [
    '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_1.csv',
    '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_2.csv',
    '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_3.csv'
]

VAL_FILE_PATH = '/F00120250015/cell_datasets/dataset_zkw/test/251011/folds_new/fold_4.csv'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Cell Classification Training')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--target_size', type=int, default=224, help='图像目标尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'adamw'], help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    
    # 学习率调度器参数
    parser.add_argument('--scheduler', type=str, default='plateau', 
                        choices=['plateau', 'step', 'cosine'], help='学习率调度器类型')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='ReduceLROnPlateau的patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR的step_size')
    
    # 数据增强参数
    parser.add_argument('--brightness', type=float, default=0.2, help='亮度调整范围')
    parser.add_argument('--contrast', type=float, default=0.2, help='对比度调整范围')
    parser.add_argument('--rotation', type=int, default=10, help='随机旋转角度')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'], help='模型架构')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练权重')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Tensorboard日志目录')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(args, train=True):
    """根据参数构建数据增强"""
    if train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(args.target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(args.rotation),
            transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(args.target_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    return transform


class cellDataset(Dataset):
    """自定义数据集类（使用均值填充）、自定义标签分配"""
    def __init__(self, dataframe, transform=None, target_size=224, num_classes=2):
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
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["image_file_path"]
        cell_type = row["cell_type"]
        
        # CHC -> 1, 白细胞 -> 0
        label = 0 if cell_type != "CHC" else 1
        
        original_img = cv2.imread(img_path)
        padded_img = self.pad_image_with_mean(original_img, self.target_size)
        img = Image.fromarray(padded_img)
        
        if self.transform:
            img = self.transform(img)
        return img, label


def load_csv(train_csv_paths, val_csv_path):
    """返回训练集和验证集的df"""
    print('\nloading training dataset...')
    train_dfs = []
    for train_path in train_csv_paths:
        df = pd.read_csv(train_path)
        train_dfs.append(df)
        print(f"{train_path}: {len(df)}个训练样本")
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    print("loading validation dataset...")
    val_df = pd.read_csv(val_csv_path)
    print(f"{val_csv_path}: {len(val_df)}个验证样本")
    
    # 只保留需要的数据列    
    required_cols = ['image_file_path', 'cell_type', 'x', 'y', 'w', 'h']
    train_df = train_df[required_cols]
    val_df = val_df[required_cols]
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"总训练样本数: {len(train_df)}")
    print("\n细胞类别分布:")
    print(train_df['cell_type'].value_counts())
    
    print("\nbinary classification (excluding CTC)...")
    train_df = train_df[train_df['cell_type'].isin(['CD66b', 'CD14', 'WBC', 'CD3', 'CHC'])]
    val_df = val_df[val_df['cell_type'].isin(['CD66b', 'CD14', 'WBC', 'CD3', 'CHC'])]
    
    print('total count of training data (CTC excluded): ', len(train_df))
    
    train_0 = train_df[train_df['cell_type'] == 'CHC']
    train_1 = train_df[train_df['cell_type'] != 'CHC']
    
    print(f"count of CHC: {len(train_0)}")
    print(f'count of WBC (including other type of wbc): {len(train_1)}')
    print("="*60 + "\n")
    
    # 展示一张图片
    # showImg(train_df)
    
    return train_df, val_df


def get_model(args):
    """根据参数创建模型"""
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    return model


def get_optimizer(args, model):
    """根据参数创建优化器"""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                             momentum=args.momentum, weight_decay=args.weight_decay)
    
    return optimizer


def get_scheduler(args, optimizer):
    """根据参数创建学习率调度器"""
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.scheduler_patience, 
            factor=args.scheduler_factor
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.scheduler_factor
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (img, labels) in enumerate(dataloader):
        img, labels = img.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() #把一个元素的张量转成python标量
        # output.shape == [B, 2]
        _, predicted = output.max(1) # 返回第二维度的最大值(置信度高的那一类),和它所属的类别predicted
        
        total += len(labels)
        correct += predicted.eq(labels).sum().item()
        
        # 记录每个batch的loss
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
        
        if (batch_idx + 1) % 50 == 0:
            batch_acc = 100. * correct / total
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Loss = {loss.item():.4f}, Acc = {batch_acc:.2f}%")
   
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, num_classes=2):
    """验证函数"""
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    class_correct = [0] * num_classes # 每一个类预测对了有多少个
    class_total = [0] * num_classes # 每一个类的总数有多少个
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += len(labels)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    class_accs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accs.append(100. * class_correct[i] / class_total[i])
        else:
            class_accs.append(0.0)
    
    return epoch_loss, epoch_acc, class_accs

def showImg(df):
    row = df.iloc[0]
    img_path = row['image_file_path']
    img = Image.open(img_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("./preview.png")

def main():
    # 解析参数
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'{args.model}_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # 记录超参数
    writer.add_text('Hyperparameters', str(vars(args)))
    
    print("="*60)
    print("训练配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # 加载数据
    train_df, val_df = load_csv(train_csv_paths=TRAIN_FILE_PATHS, val_csv_path=VAL_FILE_PATH)
    
    train_dataset = cellDataset(
        train_df, 
        transform=get_transforms(args, train=True),
        target_size=args.target_size,
        num_classes=2
    )
    val_dataset = cellDataset(
        val_df, 
        transform=get_transforms(args, train=False),
        target_size=args.target_size,
        num_classes=2
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = get_model(args)
    model.to(device)
    
    # 创建优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    
    print(f"开始训练 {args.model}...")
    print(f"总共 {args.num_epochs} 个epoch\n")
    
    best_val_acc = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        
        # 验证
        val_loss, val_acc, class_accs = validate(
            model, val_loader, criterion, device, num_classes=2
        )
        
        # 更新学习率
        if args.scheduler == 'plateau':
            scheduler.step(val_loss) # 连续patience轮val_loss没动，以*factor的速度降低学习率，以供下次训练
        else:
            scheduler.step()
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Accuracy/Class0_WBC', class_accs[0], epoch)
        writer.add_scalar('Accuracy/Class1_CHC', class_accs[1], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 打印信息
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  类别0(白细胞)准确率: {class_accs[0]:.2f}%")
        print(f"  类别1(CHC)准确率: {class_accs[1]:.2f}%")
        print(f"  当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, save_path)
            print(f"保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        
        # 定期保存checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, checkpoint_path)
            print(f"保存checkpoint: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"TensorBoard日志保存在: {log_dir}")
    print(f"运行 'tensorboard --logdir={args.log_dir}' 查看训练曲线")
    print(f"{'='*60}\n")
    
    writer.close()


if __name__ == '__main__':
    main()