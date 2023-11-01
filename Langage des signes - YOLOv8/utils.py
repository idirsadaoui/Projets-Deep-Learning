import os
import cv2
import time
import threading
from typing import Tuple
from ultralytics import YOLO

LARGEUR = 480
HAUTEUR = 640


#################################################
#              CRÉATION DU DATASET              #
#################################################

def capture_webcam_images(output_folder: str = 'captures_webcam',
                          resolution: Tuple[int, int] = (LARGEUR, HAUTEUR),
                          classe: str = 'A'):
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

PATH = "./Langage des signes - YOLOv8/weights/Langage_signe_ABCDEF_yolov8.pt"
CONF_THRESDOLD = 0.5
PATH_GIF = "./Langage des signes - YOLOv8/data"


def real_time_detections(model_path: str = PATH,
                         conf_threshold: float = CONF_THRESDOLD
                         save_gif: bool = True,
                         path_gif: str = "./Langage des signes - YOLOv8/data"):
    # Initialise la webcam avec la résolution souhaitée
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGEUR)  # Largeur de la vidéo
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HAUTEUR)  # Hauteur de la vidéo

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
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Ferme la fenêtre si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libère la webcam et ferme la fenêtre
    cap.release()
    cv2.destroyAllWindows()

    if save_gif:
      frames[0].save(os.path.join(path_gif, "LST.gif"), save_all=True, append_images=frames[1:], duration=100, loop=0)

