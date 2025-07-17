import cv2
import mediapipe as mp
import numpy as np
import requisicoes

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

ssid = requisicoes.get_ssid()

print(f"SSID conectado: {ssid}")

ARM_INDICES = [0, 1, 2, 3, 4, 5]


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
    font_scale = 1* scale_factor
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

lastMsgSent = ""
def draw_landmarks(frame, landmarks):
    global lastMsgSent
    cores = {
        'dedos': (0, 255, 0),
        'bracos': (255, 0, 0),
        'corpo': (0, 0, 255)
    }

    for finger, indices in FINGER_INDICES.items():
        for i in range(len(indices) - 1):
            start = landmarks[indices[i]]
            end = landmarks[indices[i + 1]]
            cv2.line(frame, (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                         (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])), cores['dedos'], 2)
            cv2.circle(frame, (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])), 5, cores['dedos'], -1)

    for i in range(len(ARM_INDICES) - 1):
        start = landmarks[ARM_INDICES[i]]
        end = landmarks[ARM_INDICES[i + 1]]
        cv2.line(frame, (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])),
                     (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])), cores['bracos'], 2)
        cv2.circle(frame, (int(start.x * frame.shape[1]), int(start.y * frame.shape[0])), 5, cores['bracos'], -1)

    if landmarks:
        x_min = int(min([landmarks[i].x for i in range(21)]) * frame.shape[1])
        x_max = int(max([landmarks[i].x for i in range(21)]) * frame.shape[1])
        y_min = int(min([landmarks[i].y for i in range(21)]) * frame.shape[0])
        y_max = int(max([landmarks[i].y for i in range(21)]) * frame.shape[0])
        color = (0, 255, 0) if is_hand_open(landmarks) else (0, 0, 255)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        if x_max > frame.shape[1] // 2:
            message = "http://192.168.4.1/get?input1=1"
            print(f"direita { 'aberta' if is_hand_open(landmarks) else 'fechada' }")
        else:
            message = "http://192.168.4.1/get?input1=9"
            print(f"esquerda { 'aberta' if is_hand_open(landmarks) else 'fechada' }")
        pulso = landmarks[0]
        polegar_ponta = landmarks[4]
        indicador_ponta = landmarks[8]
        meio_ponta = landmarks[12]
        
        # Vetores de direção
        delta_x = polegar_ponta.x - pulso.x
        delta_y = polegar_ponta.y - pulso.y
        
        # Determinar gesto principal
        gesto = ""
        if (True):#(not is_hand_open(landmarks)):
            if abs(delta_y) > abs(delta_x):
                if delta_y < -0.1:  # Polegar acima do pulso
                    gesto = "JOIA CIMA"
                elif delta_y > 0.1:  # Polegar abaixo do pulso
                    gesto = "JOIA BAIXO"
            else:
                if delta_x > 0.1:   # Polegar à direita do pulso
                    gesto = "JOIA DIREITA" 
                elif delta_x < -0.1:  # Polegar à esquerda do pulso
                    gesto = "JOIA ESQUERDA"

        # Estado da mão
        estado_mao = "ABERTA" if is_hand_open(landmarks) else "FECHADA"
        
        # Texto combinado
        texto_status = f"{gesto}"
        
        # Caixa de texto
        x_min = int(min([lm.x for lm in landmarks]) * frame.shape[1])
        y_min = int(min([lm.y for lm in landmarks]) * frame.shape[0])
        cv2.putText(frame, texto_status, (x_min, y_min - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Envio de comandos
        mapeamento_comandos = {
            "JOIA CIMA": "input1=1",
            "JOIA DIREITA": "input1=2",
            "JOIA ESQUERDA": "input1=8",
            "JOIA BAIXO": "input1=7"
        }
        
        comando = mapeamento_comandos.get(gesto, "input1=9")
        message = f"http://192.168.4.1/get?{comando}&estado={estado_mao.lower()}"

    if(True):#(message!=lastMsgSent):
        requisicoes.start_thread(message)
        lastMsgSent = message
    
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 2)
def is_hand_open(landmarks):
    open_fingers = 0
    for finger in ['index', 'middle', 'ring', 'pinky']:
        tip = landmarks[FINGER_INDICES[finger][-1]]
        pip = landmarks[FINGER_INDICES[finger][1]]
        if tip.y < pip.y:
            open_fingers += 1
    return open_fingers >= 3

def mouse_callback(event, x, y, flags, param):
    global texto_teste
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > 640:
            if 100 < y < 150:
                texto_teste = "Botao 1 pressionado!"
            elif 160 < y < 210:
                texto_teste = "Botao 2 pressionado!"

cv2.namedWindow('Capricornio', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Capricornio', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Capricornio', mouse_callback)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    

    hand_state = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            hand_open = is_hand_open(landmarks)
            hand_state = "ABERTA" if hand_open else "FECHADA"
            texto_teste = f"Mao detectada: {hand_state}"
            draw_landmarks(frame, landmarks)
    else:
        message = "http://192.168.4.1/get?input1=9"
        print("no landmarks")
        requisicoes.start_thread(message)
        lastMsgSent = message
    
    combined_frame = draw_gui(frame, hand_state)
    
    cv2.imshow('Capricornio', combined_frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
