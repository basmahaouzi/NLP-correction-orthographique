
import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

cap = cv2.VideoCapture(0)
#Ouvre une camera pour la capture video.
#MediaPipe Hands est une solution de suivi des mains et des doigts haute fidelite. Il utilise l'apprentissage automatique (ML) pour deduire 21 points de repere 3D d'une main a partir d'une seule image.
mpHands = mp.solutions.hands
#est utilise pour charger toutes les fonctionnalites pour effectuer la detection des mains
hands = mpHands.Hands()
# Le constructeur de la classe Hands() a des parametres facultatifs tels que static_image_mode, max_num_hands, min_detection_confidence et min_tracking_confidence.
mpDraw = mp.solutions.drawing_utils
# est utilise pour dessiner les mains detectees sur l'image.
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#Pour Obtenir le peripherique audio par defaut a l'aide de PyCAW.
volMin, volMax = volume.GetVolumeRange()[:2]
#Equivalent a :
# volRange = volume.GetVolumeRange()
# volMin = volRange[0]
# volMax = volRange[1]

# Pour but d’obtenir le volume range avec la methode GetVolumeRange() 
# la premiere case [0] represente le volume minimum et la seconde [1] represente le maximum, et on a aussi une troisieme et represente l'incrementation.(on l’a pas utilise).

while True:
    success, img = cap.read()
    #La methode cap.read() est fournie par la classe VideoCapture().
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Nous allons convertir l'image capturee par notre webcam en RGB.
    #mais pourquoi RGB ?
    #La raison pour laquelle nous faisons cela est que lorsque OpenCV a ete developpe pour la premiere fois il y a de nombreuses annees, la norme de lecture d'une image etait l'ordre BGR.
    #Au fil des ans, la norme est devenue RGB, mais OpenCV maintient toujours cet ordre BGR "herite" pour garantir l'absence de rupture de code existante.
    results = hands.process(imgRGB)
    #detecter les mains puis stocker les resultats de la detection des reperes de la main dans la variable results.
    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                #la largeur et la hauteur d'origine de l'image a partir de l'exemple d'image que nous avons defini.
                cx, cy = int(lm.x * w), int(lm.y * h)
                #Les coordonnees cx cy permet de fournir les coordonnees du point situe au centre de la  distance entre les bouts des 2doigts et les redimensionner par rapport a la taille de l’image.
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)
            #Mediapipe drawing_utils fournit une methode appelee draw_landmarks() qui nous aide a relier les points (points cles) que nous avons detectes.
            #Cet expression permet de dessiner le reperage des points sur la main detectee . 

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        #dessiner un cercle sur n'importe quelle image
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #La methode cv2.line() est utilisee pour tracer une ligne liant de points detectes de l’image.
        length = hypot(x2 - x1, y2 - y1)
        #Hypot():renvoie sqrt(a^2 + b^2) lorsque a et b sont donnés en paramètres.
        vol = np.interp(length, [15, 220], [volMin, volMax])
        #La fonction numpy.interp() renvoie l'interpolant linéaire unidimensionnel par morceaux à une fonction avec des points de données discrets donnés (xp, fp), évalués à x.
        print(vol, length)
        volume.SetMasterVolumeLevel(vol, None)
        #La méthode SetMasterVolumeLevel définit le niveau de volume principal, en décibels, du flux audio qui entre ou sort du périphérique de point de terminaison audio
        
        
        # Hand range 15 - 220
        # Volume range -63.5 - 0.0

    cv2.imshow('Image', img)
    #nous devons montrer la sortie finale, l'image finale à l'utilisateur, c'est là que la méthode cv2.imshow() est utile.
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    #Quand l’utilisateur clique sur q alors nous pouvons sortir de la boucle.

