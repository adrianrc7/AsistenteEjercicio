import cv2
import mediapipe as mp
import numpy as np

# Inicialización de Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)  # Captura de video desde la cámara
recording = False
output = None
rep_count = 0  # Contador de repeticiones
squat_position = False
pushup_position = False
curl_position = False

# Función para grabar video
def start_recording():
    global output, recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('exercise_session.avi', fourcc, 20.0, (640, 480))
    recording = True

def stop_recording():
    global output, recording
    if output:
        output.release()
    recording = False

# Función para calcular ángulo entre tres puntos
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# Función para realizar sentadillas
def squat_exercise(landmarks):
    global rep_count, squat_position
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
    knee_angle = calculate_angle(hip, knee, ankle)

    # Verificación adicional para la alineación de rodillas sobre los tobillos
    if knee[0] < hip[0] - 0.1 or knee[0] > hip[0] + 0.1:
        position_feedback = "Ajusta la alineacion de las rodillas."
    else:
        position_feedback = "Posicion correcta de las rodillas."

    if knee_angle < 100 and not squat_position:
        squat_position = True
    elif knee_angle > 160 and squat_position:
        squat_position = False
        rep_count += 1

    return knee_angle, position_feedback

# Función para realizar flexiones
def pushup_exercise(landmarks):
    global rep_count, pushup_position
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Verificación adicional para la alineación del torso
    if abs(shoulder[1] - hip[1]) > 0.1:
        position_feedback = "Manten el torso recto."
    else:
        position_feedback = "Torso alineado correctamente."

    if elbow_angle < 90 and not pushup_position:
        pushup_position = True
    elif elbow_angle > 160 and pushup_position:
        pushup_position = False
        rep_count += 1

    return elbow_angle, position_feedback

# Función para realizar curl de bíceps
def bicep_curl_exercise(landmarks):
    global rep_count, curl_position
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Verificación adicional para la posición del codo junto al torso
    if abs(elbow[0] - shoulder[0]) > 0.1:
        position_feedback = "Manten el codo junto al torso."
    else:
        position_feedback = "Codo en posicion correcta."

    if elbow_angle < 40 and not curl_position:
        curl_position = True
    elif elbow_angle > 150 and curl_position:
        curl_position = False
        rep_count += 1

    return elbow_angle, position_feedback

# Función para seleccionar ejercicio
def select_exercise():
    print("Selecciona el ejercicio a realizar:")
    print("1. Sentadillas")
    print("2. Flexiones")
    print("3. Curl de bíceps")
    choice = input("Ingresa el número de tu elección: ")
    if choice == '1':
        return 'squat'
    elif choice == '2':
        return 'pushup'
    elif choice == '3':
        return 'bicep_curl'
    else:
        print("Opción no válida.")
        return None

# Función principal
exercise = select_exercise()
if exercise:
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir la imagen a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Detección de postura
            results = pose.process(image)

            # Dibujo de los resultados
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                if exercise == 'squat':
                    angle, feedback = squat_exercise(landmarks)
                    cv2.putText(image, f'Angulo de la rodilla: {int(angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(image, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                elif exercise == 'pushup':
                    angle, feedback = pushup_exercise(landmarks)
                    cv2.putText(image, f'Angulo del codo: {int(angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(image, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                elif exercise == 'bicep_curl':
                    angle, feedback = bicep_curl_exercise(landmarks)
                    cv2.putText(image, f'Angulo del codo: {int(angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(image, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Mostrar el conteo de repeticiones
                cv2.putText(image, f'Reps: {rep_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Dibujar los puntos de referencia y conexiones
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Mostrar video
            cv2.imshow('Asistente de Ejercicio', image)

            # Grabación de video
            if recording:
                output.write(image)

            # Controles para iniciar y detener grabación
            key = cv2.waitKey(1)
            if key == ord('r'):  # Presiona 'r' para empezar a grabar
                start_recording()
            elif key == ord('s'):  # Presiona 's' para detener la grabación
                stop_recording()
            elif key == ord('q'):  # Presiona 'q' para salir
                break

# Liberar recursos
cap.release()
if recording:
    stop_recording()
cv2.destroyAllWindows()
