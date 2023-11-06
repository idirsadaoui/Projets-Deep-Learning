import json
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union
from utils_from_json2yolo import merge_multi_segment
from ultralytics import YOLO


#################################################
#     CONVERSION FORMAT COCO JSON VERS YOLO     #
#################################################

def coco_json_to_yolo(json_file: str,
                      output_path: str):

    """
    Fonction qui a pour but de convertir un fichier COCO JSON
    au format YOLO.

    Args:
        json_file (str): Chemin vers le fichier d'annotations .json.
                         (train, valid ou test)
        output_path (str): Chemin où seront stockés les labels au format
                           YOLO.
    """

    with open(json_file) as f:
        data_RR = json.load(f)

    # Création d'un dictionnaire d'images
    images = {f'{x["id"]:d}': x for x in data_RR['images']}
    # Création d'un dictionnaire d'annotations
    imgToAnns = defaultdict(list)
    for ann in data_RR['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    bboxes = []
    segments = []
    image_name = []
    # Écrire les fichiers labels
    for img_id, anns in imgToAnns.items():
        img = images[f'{img_id:d}']
        h, w, f = img['height'], img['width'], img['file_name']

        for ann in anns:
            if ann['iscrowd']:
                continue
            # Le format COCO est : [top left x, top left y, width, height]
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left vers center
            box[[0, 2]] /= w  # normalisation de x
            box[[1, 3]] /= h  # normalisation de y
            # si la lagueur <= 0 et la hauteur <= 0
            if box[2] <= 0 or box[3] <= 0:
                continue

            box = [0] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            if ann.get('segmentation') is not None:
                if len(ann['segmentation']) == 0:
                    segments.append([])
                    continue
                if len(ann['segmentation']) > 1:
                    s = merge_multi_segment(ann['segmentation'])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    # tous les segments sont concaténés
                    s = [j for i in ann['segmentation'] for j in i]
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [0] + s
                if s not in segments:
                    segments.append(s)
                image_name.append(f)

    # Parcourir chaque sous-liste et écrire un fichier pour chacune
    for (i, j) in tqdm(zip(segments, image_name)):
        # Générer un nom de fichier unique pour chaque sous-liste
        output_filename = os.path.join(output_path, f'{j[:-4]}.txt')

        # Concaténer les éléments de la sous-liste en une chaîne de caractères
        content = ' '.join(map(str, i))

        # Écrire le contenu dans le fichier
        with open(output_filename, 'w') as file:
            file.write(content)
    print("Conversion terminée ✓")


#################################################
#    VISUALISATION DES DONNÉES ET DES LABELS    #
#################################################

model_weights = "./path_to_the_model_weights"
model = YOLO(model_weights)

def Visualization(images_abs_path: Union[str, list],
                  predictions: bool = False):

    """
    La fonction a pour but de visualiser les images du jeu de données
    accompagnées de leurs labels.
    Elle permet également de visualiser des prédictions sur des images
    avec 'predictions = True'

    !ATTENTION! le format doit être le suivant :

    -    nom_projet/
         |-- train/
         |   |-- images/
         |   |-- labels/
         |-- valid/
         |   |-- images/
         |   |-- labels/
         |-- test/
         |   |-- images/
         |   |-- labels/
         
    -    Lorsque predictions est reglé sur True, le modèle doit être déjà chargé 
         dans une variable nommée 'model'.

    Args:
        images_abs_path (Union[str, list]): Chemin vers une image ou une
                                            liste de chemins d'images.
        predictions (bool): Passe la fonction en mode visualisation de prédictions.
    """

    if isinstance(images_abs_path, str):
        images_abs_path = [images_abs_path]

    images_fin = []
    titles = []

    for image_path in images_abs_path:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        mask = np.zeros_like(image)
        mask_color = (255, 0, 255)
        
        if predictions:
            image_vide = np.zeros_like(image)
            result = model.predict(image, verbose = False)  
            mask_coordinates = result[0].masks.xy[0].tolist()
            mask_coordinates = np.array(mask_coordinates, dtype=np.int32)  
            
            cv2.fillPoly(mask, [mask_coordinates], mask_color)
            mask_image = cv2.add(image_vide, mask)

            images_fin.extend([image, mask_image])
            titles.extend([image_name, 'Mask prediction'])
        else:
            image_name_2 = image_name.split('_jpg')[0]
            mask_folder_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'labels')
            mask_file_path = os.path.join(mask_folder_path, image_name + ".txt")

            with open(mask_file_path, 'r') as fichier:
                contenu_fichier = fichier.readlines()
                normalized_coordinate = list(map(float, contenu_fichier[0].split(" ")))[1:]

            height, width = mask.shape[:2]

            image_mask = cv2.fillPoly(mask,
                                    [np.array([(int(width * x), int(height * y)) for x, y in zip(*[iter(normalized_coordinate)] * 2)], dtype=np.int32)],
                                    mask_color)

            images_fin.extend([image, image_mask])
            titles.extend([image_name_2, 'Mask'])

    n_images = len(images_fin)
    if predictions:
        ncol = min(n_images, 6)
    else:
        ncol = min(n_images, 4)
    nrow = (n_images + ncol - 1) // ncol

    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 15))

    for ax, img, title in zip(axs.flatten(), images_fin, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for i in range(n_images, nrow * ncol):
        axs.flatten()[i].axis('off')

    plt.tight_layout()
    plt.show()
