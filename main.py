# -*- coding: utf-8 -*-
import sys
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn


# Эмоции
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Путь к модели
DEFAULT_MODEL_PATH = 'models/Arkhipov_F1_model.pth'


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(EmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Блок 1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}\n"
            f"Сначала надо обучить модель!"
        )
    
    # GPU если доступен, иначе CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = EmotionCNN(num_classes=7, dropout_rate=0.5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 
        return model, device
    
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {e}")

# Предобработка изображения лица
def preprocess_face(face_img):
    # Изменение размера до 48x48
    face_resized = cv2.resize(face_img, (48, 48))
    
    face_normalized = face_resized / 255.0
    face_normalized = (face_normalized - 0.5) / 0.5
    
    face_tensor = torch.FloatTensor(face_normalized).unsqueeze(0).unsqueeze(0)
    
    return face_tensor

# Предсказание эмоции
def predict_emotion(model, face_img, device):
    face_tensor = preprocess_face(face_img).to(device)
    
    with torch.no_grad():
        output = model(face_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    emotion = EMOTIONS[predicted.item()]
    conf_percent = confidence.item() * 100
    
    return emotion, conf_percent


def process_image_file(image_path, model, device, face_cascade):
    
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return False
    
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Не могу загрузить: {image_path}")
            print("  Поддерживаются: .jpg, .jpeg, .png, .bmp")
            return False
    except Exception as e:
        print(f"Ошибка: {e}")
        return False
    
    print(f"\nОбрабатываю: {image_path}")
    print(f"   Размер: {frame.shape[1]}x{frame.shape[0]} px")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("\nЛицо не найдено")
        return False
    
    print(f"\nНайдено лиц: {len(faces)}")
    print("  Результаты CNN")
    
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(model, face_roi, device)
        
        print(f"\nЛицо #{i+1}:")
        print(f"  Эмоция:      {emotion}")
        print(f"  Точность: {confidence:.1f}%")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion}: {confidence:.1f}%"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    processing_time = time.time() - start_time
    print(f"\nВремя: {processing_time:.3f} сек")
    
    print("\nНажми любую клавишу чтоб закрыть")
    cv2.imshow('Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True

# Парсинг аргументов командной строки
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Emotion Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument('-i', '--image', type=str, default=None, help='Путь к фотке')
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL_PATH, help='Путь к модели')
    
    return parser.parse_args()

# ГЛАВНАЯ ФУНКЦИЯ
def main():
    args = parse_arguments()
    
    print("РАСПОЗНАВАНИЕ ЭМОЦИЙ")
    print("by nothing_codes")
    
    # Загрузка модели
    print("\nЗагружаю нейронку...")
    print(f"   Модель: {args.model}")
    
    try:
        model, device = load_model(args.model)
        print(f"   Модель загружена")
        print(f"   Устройство: {device}")
    except FileNotFoundError as e:
        print(f"\n {e}")
        return 1
    except Exception as e:
        print(f"\nОшибка: {e}")
        return 1
    
    print("\nЗагружаю детектор лиц...")
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise RuntimeError("Не удалось загрузить детектор")
        print("Детектор загружен")
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    if args.image:
        print("\nРежим: файл")
        success = process_image_file(args.image, model, device, face_cascade)
        return 0 if success else 1
    
    print("\nРежим: веб-камера")
    return run_camera_mode(model, device, face_cascade)


def run_camera_mode(model, device, face_cascade):
    
    print("   Подключаюсь к камере...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n  Камера не работает")
        return 1
    
    print("Камера подключена")
    
    print("\nУПРАВЛЕНИЕ:")
    print("  ПРОБЕЛ - сделать фотку и распознать")
    print("  ESC    - выход\n")
    
    try:
        result_frame = None
        show_result_until = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Ошибка чтения с камеры")
                break
            
            # Показываем результат если есть
            current_time = time.time()
            if result_frame is not None and current_time < show_result_until:
                display_frame = result_frame
            else:
                display_frame = frame
                result_frame = None
            
            cv2.imshow('Camera - SPACE: photo, ESC: exit', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            # ПРОБЕЛ
            if key == 32:
                print("\nОбработка...")
                start_time = time.time()
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) == 0:
                    print("  Лицо не найдено")
                else:
                    print(f"Найдено лиц: {len(faces)}")
                    print("  Результаты")
                    
                    # Создаем копию кадра для результата
                    result_frame = frame.copy()
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        face_roi = gray[y:y+h, x:x+w]
                        emotion, confidence = predict_emotion(model, face_roi, device)
                        
                        print(f"\nЛицо #{i+1}:")
                        print(f"  Эмоция:      {emotion}")
                        print(f"  Точность: {confidence:.1f}%")
                        
                        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text = f"{emotion}: {confidence:.1f}%"
                        cv2.putText(result_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    processing_time = time.time() - start_time
                    print(f"\nВремя: {processing_time:.3f} сек\n")
                    
                    # Показываем результат 3 секунды
                    show_result_until = time.time() + 3.0
            
            # ESC
            elif key == 27:
                print("\nРабота окончена!")
                break
    
    except KeyboardInterrupt:
        print("\n\nНеожиданная остановка")
    except Exception as e:
        print(f"\nОшибка: {e}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("Завершено\n")
    return 0


# Точка входа в программу
if __name__ == '__main__':
    sys.exit(main())
