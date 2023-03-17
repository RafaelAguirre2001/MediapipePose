import cv2
import mediapipe as mp

# Inicializar los módulos de Mediapipe para la detección de postura, manos y rostros
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

# Inicializar las utilidades de dibujo de Mediapipe
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video con OpenCV
cap = cv2.VideoCapture(0)

# Configurar los modelos de detección y seguimiento de pose, manos y rostros de Mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
    mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
    mp_face.FaceDetection(min_detection_confidence=0.5) as face_detection:
    
    while True:
        # Leer el cuadro de la captura de video
        ret, frame = cap.read()
        
        # Voltear el cuadro horizontalmente para crear un efecto de espejo
        frame = cv2.flip(frame, 1)
        
        # Convertir el cuadro a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar postura, manos y rostros en el cuadro
        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)
        results_face = face_detection.process(rgb)
        
        # Dibujar los puntos de referencia de la postura en el cuadro
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # Dibujar los puntos de referencia de las manos en el cuadro
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Dibujar las detecciones de rostros en el cuadro
        if results_face.detections:
            for detection in results_face.detections:
                mp_drawing.draw_detection(frame, detection)
                
        # Mostrar el cuadro
        cv2.imshow('Detección', frame)
        
        # Salir del bucle cuando se presiona la tecla "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la captura de video y destruir las ventanas
cap.release()
cv2.destroyAllWindows()
