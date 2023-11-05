# <div align="center"> Détection de l'alphabet du langage des signes français </div>

<div align="center">
  <p>
      <img width="70%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/support/Amour.png"></a>
  </p>
  
Ce projet à pour but d'explorer le modèle [YOLOv8](https://docs.ultralytics.com/models/yolov8/#overview) proposé par [Ultralytics](https://ultralytics.com), en se concentrant plus particulièrement sur le modèle de détection d'objets.

L'idée générale est de détecter les 6 premières lettres de l'alphabet du langage des signes français dans le but d'effectuer des prédictions en temps réel avec la Webcam.
  
</div>

## 

## <div align="center">Création du jeu de données</div>

Les classes du jeu de données sont les 6 premières lettres de l'alphabet, à savoir A, B, C, D, E et F.

Le jeu de données est construit grâce à la fonction `capture_webcam_images` du fichier [utils.py](https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/utils.py).

`capture_webcam_images` prend en argument les variables suivantes :

* `output_folder` est une chaîne de caractères correspondant au nom du dossier dans lequel les données seront stockées.
* `resolution` est un tuple d'entiers correspondant aux dimensions de la webcam et, par conséquent, aux dimensions des images du jeu de données.
* `classe` est un caractère correspondant à la classe de l'image.

Cette fonction a pour but d'initialiser et d'ouvrir la webcam, puis de prendre des captures d'écran toutes les 2 secondes.
Avant chaque capture d'écran, le geste correspondant à l'une des classes est effectué.

Voici un aperçu des classes du jeu de données :

<div align="center">
  <p>
      <img width="85%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/support/Apercu_dataset.png"></a>
  </p>
</div>

Le jeu de données complet est disponible sur ce [lien](https://universe.roboflow.com/idir-sadaoui-qgzta/langage-des-signes).

##

## <div align="center">Labélisation</div>

La labélisation du jeu de données est effectuée avec [Roboflow](https://roboflow.com) en plusieurs étapes :

* Une bounding box est tracée sur chaque image autour du geste de la main.
* Un label est attribué en fonction de la lettre de l'alphabet.
* Une augmentation des données sur le jeu d'entraînement est effectuée, incluant des retournements horizontaux, des recadrages aléatoires, et l'ajout de bruit.
* Un redimensionnement des images en 640x640 pixels est effectué pour faciliter l'entraînement du modèle YOLO.

Référez-vous au lien du jeu de données complet pour obtenir davantage de précisions.

Voici la répartition des classes après labélisation (sans augmentation de données) :

<div align="center">
  <p>
      <img width="45%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/support/labels_repartition.jpg"></a>
  </p>
</div>

## 

## <div align="center">Entraînement</div>

Le jeu de données est téléchargé au format YOLOv8 sur Roboflow et se présente de la façon suivante :


```lua
Langage_des_signes_Roboflow/
|-- train/
|   |-- images/
|       |-- capture_20231003-144600_A.png
|       |-- capture_20231003-144601_A.png
|       |-- ...
|   |-- labels/
|       |-- capture_20231003-144600_A.txt
|       |-- capture_20231003-144601_A.txt
|       |-- ...
|-- valid/
|   |-- images/
|   |-- labels/
|-- test/
|   |-- images/
|   |-- labels/
|-- data.yaml
```

L'entraînement est effectué sur Google Colab avec un GPU T4, en utilisant comme point de départ les poids du modèle `yolov8m.pt` disponibles dans le dossier `yolov8` en utilisant le code suivant :

#### code

```python
!yolo train data = ./data.yaml model=yolov8m.pt epochs = 100
```

Voici les résultats de l'entraînement pour 100 epochs et la matrice de confusion du jeu de données de test :

<div align="center">
  <p>
      <img width="95%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/support/resultats_entrainement.png"></a>
  </p>
</div>

##

## <div align="center"> Prédictions en temps réel </div>

La prédiction en temps réel est basée sur la fonction `real_time_detections` du fichier [utils.py](https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/utils.py).

`real_time_detections` prend en argument les variables suivantes :

* `model_path` est une chaîne de caractères correspondant au chemin du fichier de poids du modèle.
* `conf_threshold` est un nombre compris entre 0 et 1, représentant le seuil de confiance minimum attendu pour les prédictions.
* `save_gif` est un booléen servant à sauvegarder la vidéo au format gif.
* `path_gif` est une chaîne de caractères correspondant au chemin dans lequel sera sauvegardé le gif.

Cette fonction a pour but d'initialiser et d'ouvrir la webcam, puis d'effectuer des prédictions et détections pour les 6 premières lettres de l'alphabet du langage des signes français.

#### Utilisation de la fonction `real_time_detection`

```python
real_time_prediction(model_path = "./weights/Langage_signe_ABCDEF_yolov8.pt",
                     conf_threshold = 0.5,
                     save_gif = True,
                     path_gif = "./support")
```

#### Prédictions effectuées avec une webcam

<div align="center">
  <p>
      <img width="60%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/support/LSF.GIF"></a>
  </p>
</div>

