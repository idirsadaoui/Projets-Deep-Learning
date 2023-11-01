import os
import cv2
import time
import threading
from typing import Tuple
from ultralytics import YOLO
from PIL import Image

#################################################
#              CRÉATION DU DATASET              #
#################################################


def capture_webcam_images(output_folder: str = 'captures_webcam',
                          resolution: Tuple[int, int] = (640, 480),
                          classe: str = 'A'):

    """
    Fonction qui a pour but d'initialiser et d'ouvrir la webcam,
    puis de prendre des captures d'écran toutes les 2 secondes.

    Args:
        output_folder (str): Nom du dossier dans lequel les données seront stockées.
        resolution (Tuple[int, int]): Dimensions de la webcam et, par conséquent,
                                      aux dimensions des images du jeu de données.
        classe (str): La classe de l'image.
    """

    # Initialise la webcam avec la résolution souhaitée
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Largeur de la vidéo
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Hauteur de la vidéo

    # Crée un dossier pour stocker les captures d'écran
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fonction pour capturer une image et la sauvegarder
    def capture_image():
        while True:
            # Capture une image de la webcam
            ret, frame = cap.read()

            # Crée un nom de fichier unique basé sur la date et l'heure
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # Le nom de l'image contient la classe (A, B, C, D, E ou F)
            image_filename = os.path.join(output_folder,
                                          f'capture_{timestamp}_{classe}.png')

            # Enregistre l'image dans le dossier spécifié
            cv2.imwrite(image_filename, frame)

            # Attend 2 secondes avant de prendre la prochaine capture
            time.sleep(2)

    # Lancez la fonction de capture d'image dans un thread séparé
    capture_thread = threading.Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    while True:
        # Affiche l'image dans une fenêtre
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)

        # Ferme la fenêtre si la touche 'q' est pressée
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Libère la webcam et ferme la fenêtre
    cap.release()
    cv2.destroyAllWindows()


#################################################
#      PRÉDICTION EN DIRECT SUR UNE WEBCAM      #
#################################################

def real_time_detections(model_path: str,
                         conf_threshold: float,
                         save_gif: bool,
                         path_gif: str):

    """
    Fonction qui a pour but a pour but d'initialiser et d'ouvrir la webcam,
    puis d'effectuer des prédictions et détectionspour les 6 premières
    lettres de l'alphabet du langage des signes français.

    Args:
        model_path (str): Chemin du fichier de poids du modèle.
        conf_threshold (float): Nombre compris entre 0 et 1, représentant le seuil
                                de confiance minimum attendu pour les prédictions.
        save_gif (bool): Booléen servant à sauvegarder la vidéo au format gif.
        path_gif (str): Chemin dans lequel sera sauvegardé le gif.
    """

    # Initialise la webcam avec la résolution souhaitée
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largeur de la vidéo
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Hauteur de la vidéo

    # Charge le modèle grâce au fichier poids .pt ayant les meilleurs
    # performances et résultant de l'entraînement
    model = YOLO(model_path)

    frames = []

    while True:
        # Affiche l'image dans une fenêtre
        ret, frame = cap.read()
        result = model(frame)

        for i in result:
            boxes = i.boxes
            for box in boxes:
                conf = box.conf.item()
                # Trace une bounding box seulement si la prédiction
                # a au moins 50% de score de confiance
                if conf > conf_threshold:
                    label = model.names[box.cls.item()]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame,
                                f'{label}: {conf:.2f}',
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2)

        cv2.imshow("Yolov8", frame)

        if save_gif:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # Ferme la fenêtre si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libère la webcam et ferme la fenêtre
    cap.release()
    cv2.destroyAllWindows()

    if save_gif:
        frames[0].save(os.path.join(path_gif, "LST.gif"), save_all=True, append_images=frames[1:], duration=100, loop=0)
