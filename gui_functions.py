import os
import csv
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


# przycięcie obrazu
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


# dokonanie predykcji do pliku csv z użyciem konkretnego modelu na zdefiniownych plikach źródłowych
def make_predictions(model_path, input_path, output_path):

    # transformacja
    if 'efficientnet_b5' in model_path:
        transform = transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # wczytanie modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model.to(device)
    model.eval()
        
    # predykcja na jednym obrazie
    def predict_one_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_cropped = crop_image_from_gray(image, tol=7)
        image_pil = Image.fromarray(image_cropped)
        image = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        return predicted.item(), probs.squeeze().cpu().numpy()

    # pętla po wszystkich obrazach
    files = []
    predictions = []
    probability = []
    probabilities = []

    # obsługa pojedynczego zdjęcia
    if os.path.isdir(input_path):
        for image_name in os.listdir(input_path):
            image_path = os.path.join(input_path, image_name)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):

                predicted_label, probs = predict_one_image(image_path)

                # print(image_name, " ", predicted_label, " ", max(probs), " ", probs)

                files.append(image_name)
                predictions.append(predicted_label)
                probability.append(max(probs))
                probabilities.append(probs)
    else:
        image_path = input_path
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):

            image_name = os.path.basename(input_path)

            predicted_label, probs = predict_one_image(image_path)

            # print(image_name, " ", predicted_label, " ", max(probs), " ", probs)

            files.append(image_name)
            predictions.append(predicted_label)
            probability.append(max(probs))
            probabilities.append(probs)

    # zapis predykcji
    with open(f"{output_path}", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Nazwa pliku', 'Diagnoza', 'Prawdopodobieństwo', 'Prawdopodobieństwa klas'])
        
        for i in range(len(files)):
            writer.writerow([
                files[i],
                predictions[i],
                probability[i],
                probabilities[i]
            ])
