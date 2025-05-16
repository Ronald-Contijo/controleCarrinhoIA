import cv2
import mediapipe as mp
import numpy as np
import requisicoes


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands()
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

ssid = requisicoes.get_ssid()
print(f"SSID conectado: {ssid}")
lastMsgSent = ""

screen_width, screen_height = 1920, 1080
PANEL_RATIO = 0.4

FINGER_INDICES = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

texto_teste = "Estado inicial"

def calculate_wrist_angle(pose_landmarks, hand_landmarks):
    if pose_landmarks is None or hand_landmarks is None:
        return None

    elbow = pose_landmarks[13]
    wrist = pose_landmarks[15]
    hand_base = hand_landmarks[0]

    # Coordenadas normalizadas (entre 0 e 1)
    a = np.array([elbow.x, elbow.y])
    b = np.array([wrist.x, wrist.y])
    c = np.array([hand_base.x, hand_base.y])

    # Vetores: antebraço e mão
    ba = a - b
    bc = c - b

    # Ângulo entre os vetores
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def draw_gui(frame, hand_state):
    h, w = frame.shape[:2]
    panel_width = int(w * PANEL_RATIO)*2
    main_width = screen_width - panel_width
    main_height = screen_height
    
    frame = cv2.resize(frame, (main_width, main_height), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((main_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    
    scale_factor = screen_height / 1080
    button_height = int(50 * scale_factor)
    font_scale = 1 * scale_factor
    
    cv2.putText(panel, f"Conectado a {ssid}", (20, int(60 * scale_factor)), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 2)
    
    btn_y = int(100 * scale_factor)
    cv2.rectangle(panel, (10, btn_y), (panel_width-10, btn_y+button_height), (0, 0, 200), -1)
    cv2.putText(panel, "Botao 1", (20, btn_y + int(35 * scale_factor)), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
    
    btn_y += int(70 * scale_factor)
    cv2.rectangle(panel, (10, btn_y), (panel_width-10, btn_y+button_height), (0, 100, 0), -1)
    cv2.putText(panel, "Botao 2", (20, btn_y + int(35 * scale_factor)), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
    
    cv2.putText(panel, texto_teste, (10, int(800 * scale_factor)), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.7, (0, 0, 255), 1)
    
    return cv2.hconcat([frame, panel])

def draw_arm_landmarks(frame, landmarks):
    ARM_CONNECTIONS = [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16)
    ]
    color = (255, 255, 0)
    
    for connection in ARM_CONNECTIONS:
        start_idx, end_idx = connection
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        
        if start.visibility > 0.5 and end.visibility > 0.5:
            start_x = int(start.x * frame.shape[1])
            start_y = int(start.y * frame.shape[0])
            end_x = int(end.x * frame.shape[1])
            end_y = int(end.y * frame.shape[0])
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
            cv2.circle(frame, (start_x, start_y), 5, color, -1)
            cv2.circle(frame, (end_x, end_y), 5, color, -1)

def calculate_elbow_angle(landmarks):
    shoulder = landmarks[11]  
    elbow = landmarks[13]     
    wrist = landmarks[15]     

    shoulder_x = shoulder.x
    shoulder_y = shoulder.y
    elbow_x = elbow.x
    elbow_y = elbow.y
    wrist_x = wrist.x
    wrist_y = wrist.y

    angle = np.arctan2(wrist_y - elbow_y, wrist_x - elbow_x) - np.arctan2(shoulder_y - elbow_y, shoulder_x - elbow_x)
    angle = np.abs(angle * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def calculate_shoulder_angle(landmarks):
    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]

    shoulder_x = shoulder.x
    shoulder_y = shoulder.y
    elbow_x = elbow.x
    elbow_y = elbow.y

    angle = np.arctan2(elbow_y - shoulder_y, elbow_x - shoulder_x) * 180.0 / np.pi
    return angle

def draw_hand_landmarks(frame, landmarks):
    global lastMsgSent

    cores = {
        'dedos': (0, 255, 0),
        'corpo': (0, 0, 255)
    }

    for finger, indices in FINGER_INDICES.items():
        for i in range(len(indices) - 1):
            start = landmarks[indices[i]]
            end = landmarks[indices[i + 1]]
            cv2.line(frame, 
                    (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                    (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])),
                    cores['dedos'], 2)
            cv2.circle(frame, 
                      (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                      5, cores['dedos'], -1)

    if landmarks:
        x_min = int(min([landmarks[i].x for i in range(21)]) * frame.shape[1])
        x_max = int(max([landmarks[i].x for i in range(21)]) * frame.shape[1])
        y_min = int(min([landmarks[i].y for i in range(21)]) * frame.shape[0])
        y_max = int(max([landmarks[i].y for i in range(21)]) * frame.shape[0])
        
        color = (0, 255, 0) if is_hand_open(landmarks) else (0, 0, 255)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        pulso = landmarks[0]
        polegar_ponta = landmarks[4]
        
        delta_x = polegar_ponta.x - pulso.x
        delta_y = polegar_ponta.y - pulso.y
        
        gesto = ""
        if abs(delta_y) > abs(delta_x):
            if delta_y < -0.1:
                gesto = "JOIA CIMA"
            elif delta_y > 0.1:
                gesto = "JOIA BAIXO"
        else:
            if delta_x > 0.1:
                gesto = "JOIA DIREITA" 
            elif delta_x < -0.1:
                gesto = "JOIA ESQUERDA"

        estado_mao = "ABERTA" if is_hand_open(landmarks) else "FECHADA"
        texto_status = f"{gesto}"
        
        cv2.putText(frame, texto_status, (x_min, y_min - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        mapeamento_comandos = {
            "JOIA CIMA": "input1=1",
            "JOIA DIREITA": "input1=a",
            "JOIA ESQUERDA": "input1=b",
            "JOIA BAIXO": "input1=7"
        }
        
        comando = mapeamento_comandos.get(gesto, "input1=9")
        message = f"http://192.168.4.1/get?{comando}&estado={estado_mao.lower()}"
        
        if message != lastMsgSent:
            requisicoes.start_thread(message)
            lastMsgSent = message

def is_hand_open(landmarks):
    open_fingers = 0
    for finger in ['index', 'middle', 'ring', 'pinky']:
        tip = landmarks[FINGER_INDICES[finger][-1]]
        pip = landmarks[FINGER_INDICES[finger][1]]
        if tip.y < pip.y:
            open_fingers += 1
    return open_fingers >= 3

cv2.namedWindow('Capricornio', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Capricornio', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)
    
    hand_state = None
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            hand_state = "ABERTA" if is_hand_open(landmarks) else "FECHADA"
            texto_teste = f"Mão detectada: {hand_state}"
            draw_hand_landmarks(frame, landmarks)
    else:
        message = "http://192.168.4.1/get?input1=9"
        requisicoes.start_thread(message)
        lastMsgSent = message
    
    if pose_results.pose_landmarks:
        draw_arm_landmarks(frame, pose_results.pose_landmarks.landmark)
        elbow_angle = calculate_elbow_angle(pose_results.pose_landmarks.landmark)
        shoulder_angle = calculate_shoulder_angle(pose_results.pose_landmarks.landmark)
        cv2.putText(frame, f"Ângulo do Cotovelo: {int(elbow_angle)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Ângulo do Ombro: {int(shoulder_angle)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
        wrist_angle = calculate_wrist_angle(
            pose_results.pose_landmarks.landmark,
            hand_results.multi_hand_landmarks[0].landmark
        )
        if wrist_angle:
            cv2.putText(frame, f"Ângulo do Pulso: {int(wrist_angle)}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    combined_frame = draw_gui(frame, hand_state)
    cv2.imshow('Capricornio', combined_frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

