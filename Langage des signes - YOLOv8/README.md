# Langage des Signes - Détection de l'alphabet

<div align="center">
  <p>
      <img width="70%" src="https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/data/Amour.png"></a>
  </p>
  
Ce projet à pour but d'explorer le modèle [YOLOv8](https://docs.ultralytics.com/models/yolov8/#overview) proposé par [Ultralytics](https://ultralytics.com), , en se concentrant plus particulièrement sur le modèle de détection d'objets.

L'idée générale est de détecter les 6 premières lettres de l'alphabet dans le but d'effectuer des prédictions en temps réel avec la Webcam.
  
</div>


## <div align="center">Création du jeu de données</div>

Les classes du jeu de données sont les 6 premières lettres de l'alphabet, à savoir A, B, C, D, E et F.

Le jeu de données est construit grâce à la fonction `capture_webcam_images` du fichier [utils.py](https://github.com/idirsadaoui/Projets-Deep-Learning/blob/main/Langage%20des%20signes%20-%20YOLOv8/utils.py)

`capture_webcam_images` prend en argument les variables suivantes :

* `output_folder` est une chaîne de caractères correspondant au nom du dossier dans lequel les données seront stockées.
* `resolution` est un tuple d'entiers correspondant aux dimensions de la webcam et, par conséquent, aux dimensions des images du jeu de données.
* `classe` est un caractère correspondant à la classe de l'image.

Cette fonction a pour but d'initialiser et d'ouvrir la webcam, puis de prendre des captures d'écran toutes les 2 secondes.
Avant chaque capture d'écran, le geste correspondant à l'une des classes est effectué.

Voici un aperçu des classes du jeu de données :

<div align="center">
  <p>
      <img width="80%" src=""></a>
  </p>
</div>


