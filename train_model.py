import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Дефолтные настройки
DEFAULT_TRAIN_DIR = "train"
DEFAULT_TEST_DIR = "test"
IMG_SIZE = 48
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS_BASE = 20
DEFAULT_EPOCHS_OPTIMIZED = 30
DEFAULT_LEARNING_RATE = 0.001
NUM_WORKERS = 4

# Я
PARTICIPANT_NAME = "Arkhipov"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Обучение нейросети',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument('--train_dir', type=str, default=DEFAULT_TRAIN_DIR, help='Папка train')
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DIR, help='Папка test')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS_OPTIMIZED, help='Эпох для оптимизированной')
    parser.add_argument('--epochs_base', type=int, default=DEFAULT_EPOCHS_BASE, help='Эпох для базовой')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Размер батча')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--base_only', action='store_true', help='Только базовая модель')
    parser.add_argument('--output_dir', type=str, default='models', help='Папка для моделей')
    
    return parser.parse_args()


# FER-2013
class EmotionDataset(Dataset):
    """Датасет для загрузки фоток эмоций"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Загружаем пути к фоткам
        for idx, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"[!] Папка не найдена: {emotion_dir}")
                continue
            
            img_files = [f for f in os.listdir(emotion_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_name in img_files:
                self.images.append(os.path.join(emotion_dir, img_name))
                self.labels.append(idx)
        
        print(f"[+] Загружено {len(self.images)} фоток из {root_dir}")
        
        # Статистика
        unique, counts = np.unique(self.labels, return_counts=True)
        for emotion_idx, count in zip(unique, counts):
            print(f"  {EMOTIONS[emotion_idx]}: {count} шт")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('L')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"[!] Не удалось загрузить {self.images[idx]}: {e}")
            return torch.zeros(1, IMG_SIZE, IMG_SIZE), self.labels[idx]


# Архитектура
class BaseCNN(nn.Module):
    """Базовая CNN для распознавания эмоций"""
    
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(BaseCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Блок 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Блок 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Блок 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Блок 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Трансформация
def get_transforms(augment=False):
    """Трансформации для фоток"""
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


# обучение
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Обучение на первом круге"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Обучение", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision для GPU
        if scaler is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# Оценка
def evaluate_model(model, test_loader, device):
    """Оценка модели"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Оценка", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# Матрица и График
def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    """Рисует матрицу ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cbar_kws={'label': 'Количество'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Сохранено: {save_path}")


def plot_roc_curves(y_true, y_probs, save_path, title='ROC Curves'):
    """Рисует ROC-кривые"""
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    plt.figure(figsize=(12, 10))
    
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{EMOTIONS[i].capitalize()} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Случайный')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Сохранено: {save_path}")


def print_metrics(y_true, y_pred, model_name):
    """Выводит метрики"""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*70}")
    print(f"  МЕТРИКИ: {model_name}")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-score: {f1:.4f}")
    print(f"{'='*70}")
    
    print("\nДетально по классам:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4))
    
    return accuracy, f1


# ОСНОВНАЯ ФУНКЦИЯ
def main():
    """Главная функция обучения"""
    args = parse_arguments()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("  ОБУЧЕНИЕ НЕЙРОНКИ")
    print("  by nothing_codes")
    print("="*70)
    print(f"Устройство: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {'Да' if torch.cuda.is_available() else 'Нет'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nПараметры:")
    print(f"  Train: {args.train_dir}")
    print(f"  Test:  {args.test_dir}")
    print(f"  Batch: {args.batch_size}")
    print(f"  LR:    {args.lr}")
    print(f"  Эпох:  {args.epochs_base} (базовая), {args.epochs if not args.base_only else 'пропуск'} (оптимизированная)")
    print("="*70)
    
    # Проверка папок
    if not os.path.exists(args.train_dir):
        print(f"\n✗ Папка train не найдена: {args.train_dir}")
        return 1
    if not os.path.exists(args.test_dir):
        print(f"\n✗ Папка test не найдена: {args.test_dir}")
        return 1
    
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========== ЗАГРУЗКА ДАННЫХ ==========
    print("\n" + "="*70)
    print("  ЗАГРУЗКА ДАННЫХ")
    print("="*70)
    
    try:
        train_dataset = EmotionDataset(args.train_dir, transform=get_transforms(augment=False))
        test_dataset = EmotionDataset(args.test_dir, transform=get_transforms(augment=False))
    except Exception as e:
        print(f"\n✗ Ошибка загрузки: {e}")
        return 1
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False,
                             persistent_workers=True if NUM_WORKERS > 0 else False)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False,
                            persistent_workers=True if NUM_WORKERS > 0 else False)
    
    print(f"\n[+] Train: {len(train_dataset)} фоток")
    print(f"[+] Test:  {len(test_dataset)} фоток")
    
    # ========== БАЗОВАЯ МОДЕЛЬ ==========
    print("\n" + "="*70)
    print("  БАЗОВАЯ МОДЕЛЬ")
    print("="*70)
    
    base_model = BaseCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print(f"\n[+] Начинаю обучение ({args.epochs_base} эпох)...")
    
    for epoch in range(args.epochs_base):
        print(f"\nЭпоха {epoch+1}/{args.epochs_base}")
        train_loss, train_acc = train_epoch(base_model, train_loader, criterion, optimizer, DEVICE, scaler)
        print(f"  Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    
    print("\n[+] Оценка базовой модели...")
    y_pred_base, y_true_base, y_probs_base = evaluate_model(base_model, test_loader, DEVICE)
    base_accuracy, base_f1 = print_metrics(y_true_base, y_pred_base, "БАЗОВАЯ")
    
    base_model_path = os.path.join(args.output_dir, f'{PARTICIPANT_NAME}_base_model.pth')
    torch.save(base_model.state_dict(), base_model_path)
    print(f"\n[+] Сохранено: {base_model_path}")
    
    plot_confusion_matrix(y_true_base, y_pred_base, 
                         os.path.join(args.output_dir, 'base_confusion_matrix.png'),
                         'Confusion Matrix - Базовая')
    plot_roc_curves(y_true_base, y_probs_base, 
                   os.path.join(args.output_dir, 'base_roc_curves.png'),
                   'ROC - Базовая')
    
    if args.base_only:
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("  ГОТОВО")
        print("="*70)
        print(f"\nВремя: {elapsed/60:.2f} мин")
        print(f"Accuracy: {base_accuracy:.4f} ({base_accuracy*100:.2f}%)")
        print(f"F1-score: {base_f1:.4f}")
        print("="*70)
        return 0
    
    # ========== ОПТИМИЗИРОВАННАЯ МОДЕЛЬ ==========
    print("\n" + "="*70)
    print("  ОПТИМИЗИРОВАННАЯ МОДЕЛЬ (С АУГМЕНТАЦИЕЙ)")
    print("="*70)
    
    train_dataset_aug = EmotionDataset(args.train_dir, transform=get_transforms(augment=True))
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False,
                                 persistent_workers=True if NUM_WORKERS > 0 else False)
    
    opt_model = BaseCNN(num_classes=NUM_CLASSES, dropout_rate=0.5).to(DEVICE)
    optimizer_opt = optim.AdamW(opt_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_opt, mode='max', patience=5, factor=0.5, verbose=True)
    
    print(f"\n[+] Начинаю обучение ({args.epochs} эпох)...")
    
    best_acc = 0.0
    patience_counter = 0
    patience_limit = 10
    opt_model_path = os.path.join(args.output_dir, f'{PARTICIPANT_NAME}_best_model.pth')
    
    for epoch in range(args.epochs):
        print(f"\nЭпоха {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(opt_model, train_loader_aug, criterion, optimizer_opt, DEVICE, scaler)
        print(f"  Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        y_pred_val, y_true_val, _ = evaluate_model(opt_model, test_loader, DEVICE)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        scheduler.step(val_acc)
        
        print(f"  Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(opt_model.state_dict(), opt_model_path)
            print(f"  ✓ Лучшая модель! Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  Без улучшений. Patience: {patience_counter}/{patience_limit}")
        
        if patience_counter >= patience_limit:
            print(f"\n[+] Early stopping на эпохе {epoch+1}")
            break
    
    opt_model.load_state_dict(torch.load(opt_model_path))
    
    print("\n[+] Оценка оптимизированной модели...")
    y_pred_opt, y_true_opt, y_probs_opt = evaluate_model(opt_model, test_loader, DEVICE)
    opt_accuracy, opt_f1 = print_metrics(y_true_opt, y_pred_opt, "ОПТИМИЗИРОВАННАЯ")
    
    plot_confusion_matrix(y_true_opt, y_pred_opt, 
                         os.path.join(args.output_dir, 'optimized_confusion_matrix.png'),
                         'Confusion Matrix - Оптимизированная')
    plot_roc_curves(y_true_opt, y_probs_opt, 
                   os.path.join(args.output_dir, 'optimized_roc_curves.png'),
                   'ROC - Оптимизированная')
    
    # Копия для совместимости
    f1_model_path = os.path.join(args.output_dir, f'{PARTICIPANT_NAME}_F1_model.pth')
    torch.save(opt_model.state_dict(), f1_model_path)
    
    # ========== ИТОГ ==========
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("  ИТОГ")
    print("="*70)
    print(f"\nВремя: {elapsed/60:.2f} мин")
    print(f"\nБазовая:")
    print(f"  Accuracy: {base_accuracy:.4f} ({base_accuracy*100:.2f}%)")
    print(f"  F1: {base_f1:.4f}")
    print(f"\nОптимизированная:")
    print(f"  Accuracy: {opt_accuracy:.4f} ({opt_accuracy*100:.2f}%)")
    print(f"  F1: {opt_f1:.4f}")
    print(f"\nУлучшение:")
    print(f"  Accuracy: +{(opt_accuracy - base_accuracy)*100:.2f}%")
    print(f"  F1: +{(opt_f1 - base_f1):.4f}")
    print(f"\n{'='*70}")
    print("  ГОТОВО!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
