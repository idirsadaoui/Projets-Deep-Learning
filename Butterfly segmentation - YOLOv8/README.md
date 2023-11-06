# <div align="center"> Butterfly Segmentation </div>

<div align="center">
  <p>
      <img width="75%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Butterfly%20segmentation%20-%20YOLOv8/support/Butterfly_pres.png"></a>
  </p>
  <p>
    Images générées par <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a>
  </p>
  <p>
    prompt :  <i>"A Butterfly with VanGogh/Matisse/Picasso/Cézanne style"</i>
  </p>

Ce projet à pour but d'explorer le modèle [YOLOv8](https://docs.ultralytics.com/models/yolov8/#overview) proposé par [Ultralytics](https://ultralytics.com), en se concentrant plus particulièrement sur le modèle de segmentation sémantique.

L'idée générale est de segmenter des images de papillons afin de prédire les masques associés.

</div>

##

## <div align="center">Jeu de données et annotations</div>

Le jeu de données a été mis à disposition par [phucthaiv02](https://www.kaggle.com/phucthaiv02) sur Kaggle.

Il est constitué de plus de 9200 images de papillons au format (224x224), réparties en 75 classes, parmi lesquelles 550 ont été sélectionnées pour ce projet.

Les annotations du jeu de données ont été effectuées avec [Roboflow](https://roboflow.com) en deux étapes :

* Une mask est tracé sur chaque image autour du papillon.
* Une augmentation des données sur le jeu d'entraînement est effectuée,  incluant des retournements horizontaux et verticaux, des rotations aléatoires, des recadrages aléatoires, l'ajout de bruit, ainsi que des modifications de teintes sur les images.

Pour plus de précision, le jeu de données complet est accessible via ce [lien](https://universe.roboflow.com/idir-sadaoui-qgzta/butterfly-smcqg/dataset/1).

Le jeu de données est téléchargé au format COCO JSON.

### Conversion COCO JSON vers YOLO

Une conversion des annotations (en format .json) est nécessaire pour l'entraînement du modèle.

La fonction `coco_json_to_yolo` du fichier [utils.py](https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Butterfly%20segmentation%20-%20YOLOv8/utils.py) à été construite pour accomplir cette tâche.

`coco_json_to_yolo` prend en argument les variables suviantes :

* `json_file` est une chaîne de caractère correspondant au chemin vers le fichier d'annotations `.json`.
* `output_path` est une chaîne de caractère correspondant au chemin où seront stockés les labels au format YOLO.

Cette fonction a pour but de convertir les fichiers d'annotations au format `.json` téléchargés depuis Roboflow en fichiers `.txt` compatibles avec le format YOLO.

Format initial (Roboflow) :

```hue
Butterfly_segmentation_Roboflow/
|-- train/
|   |-- train_annotations.coco.json
|   |-- images/
|       |-- Image_100.jpg
|       |-- Image_101.jpg
|       |-- ...
|-- valid/
|   |-- valid_annotations.coco.json
|   |-- images/
|       |-- Image_205.jpg
|       |-- Image_213.jpg
|       |-- ...
```

Format cible (YOLO) :

```hue
Butterfly_segmentation_Roboflow/
|-- train/
|   |-- images/
|       |-- Image_100.jpg
|       |-- Image_101.jpg
|       |-- ...
|   |-- labels/
|       |-- Image_100.txt
|       |-- Image_101.txt
|       |-- ...
|-- valid/
|   |-- images/
|   |-- labels/
|-- data.yaml
```
`coco_json_to_yolo` est appliquée aux annotations des jeux d'entraînement et de validation, avec la création préalable d'un dossier labels pour chacun des deux ensembles.

De plus, le fichier `data.yaml` est créé avec les instructions nécessaires pour assurer le bon fonctionnement de la phase d'entraînement.

### Visualisation des données

La visualisation des données se fait grâce à la fonction `Visualization` du fichier [utils.py](https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Butterfly%20segmentation%20-%20YOLOv8/utils.py).

`Visualization` prend en argument les variables suviantes :

* `images_abs_path` est une chaîne de caractère ou une liste de chaînes de caractères correspondant au chemin des images.
* `predictions` est un booléen servant à visualiser les prédictions sur des images.

Cette fonction a pour but de visualiser les images du jeu de données avec leur masque associé ou le masque prédit.

#### Utilisation de `Visualization` pour la visualisation du jeu de données :

```python
from utils import Visualization

images_list = [./train/images/Images_100.jpg,
               ./train/images/Images_101.jpg,
               ./train/images/Images_102.jpg,
               ./train/images/Images_103.jpg]

Visualization(images_abs_path = images_list)
```

#### Résultats

<div align="center">
  <p>
      <img width="90%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Butterfly%20segmentation%20-%20YOLOv8/support/visualisation_dataset.png"></a>
  </p>
</div>

##

## <div align="center">Entraînement</div>

L'entraînement est effectué sur Google Colab avec un GPU T4, en utilisant comme point de départ les poids du modèle `yolo8n-seg.pt` disponibles dans le dossier `weights` en utilisant le code suivant :

#### code

```python
from ultralytics import YOLO

model = YOLO("./weights/yolo8n-seg.pt")

results = model.train(data='./data.yaml', epochs=400, imgsz=224)
```
Entraînement arrêté à l'epoch 128, car pas d'amélioration notable durant les 50 epochs précedentes.

##

## <div align="center">Prédictions</div>

La prédiction est réalisée par la fonction `Visualization` avec l'argument `predictions = True`.

Le modèle doit être importer au préalable dans une variable nommée `model` avant de lancer les prédictions avec `Visualization`.

#### Utilisation de `Visualization` pour la prédiction : 

```python
model = YOLO("./weights/butterfly_seg.pt")

images_list = [./test/Images_100.jpg,
               ./test/Images_101.jpg,
               ./test/Images_102.jpg,
               ./test/Images_103.jpg]

Visualization(images_abs_path = images_list
              predictions = True)
```

