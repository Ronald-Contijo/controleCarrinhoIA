import cv2
import mediapipe as mp
import numpy as np
import math
import time
import requisicoes

# ============== CONFIG EXTRAS (tela cheia e logo) ==============
LOGO_PATH = "logo.png"            # <- caminho da sua logo (PNG com transparência, de preferência)
LOGO_SCREEN_WIDTH_FRAC = 0.13    # fração da largura da tela ocupada pela logo (ex.: 12%)
LOGO_MARGIN_PX = 24               # margem da logo em relação às bordas
WINDOW_NAME = "Capricornio - Base direita | Manipulador esquerda (4 quadrantes)"
# ===============================================================

# ================== CONFIG EXISTENTE ==================
ROBOT_IP = "192.168.4.1"
NEUTRO_BASE = "input1=9"
JOINTS_MIN, JOINTS_MAX = 0, 180
JOINT_STEP = 3            # passo por gesto
SEND_DEBOUNCE_MS = 60     # intervalo mínimo entre envios de juntas
# =======================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def http_send(q):
    requisicoes.start_thread(f"http://{ROBOT_IP}/get?{q}")

def clamp(v,a,b): return max(a,min(b,v))

def gesture_thumb_ud(landmarks):
    """Retorna 'UP', 'DOWN' ou None com base no polegar vs pulso."""
    wrist = landmarks[0]
    thumb = landmarks[4]
    dy = thumb.y - wrist.y
    if abs(dy) < 0.06:  # zona morta
        return None
    return "UP" if dy < 0 else "DOWN"

def gesture_thumb_lr(landmarks):
    wrist = landmarks[0]
    thumb = landmarks[4]
    dx = thumb.x - wrist.x
    if abs(dx) < 0.06:
        return None
    return "RIGHT" if dx > 0 else "LEFT"

def base_cmd_from_dirs(updown, leftright):
    # prioridade vertical; se não houver, usa horizontal
    if updown == "UP": return "input1=1"   # frente
    if updown == "DOWN": return "input1=7" # ré
    if leftright == "RIGHT": return "input1=2"  # giro dir
    if leftright == "LEFT":  return "input1=8"  # giro esq
    return NEUTRO_BASE

def build_joints_payload(J):
    return f"input1=j{int(J[0])},{int(J[1])},{int(J[2])},{int(J[3])}"

# ---------- util: resolução da tela (cross-platform) ----------
def get_screen_resolution():
    # tenta via Tkinter
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        return sw, sh
    except Exception:
        # fallback comum
        return 1920, 1080

# ---------- util: letterbox mantendo aspecto ----------
def resize_letterbox(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # canvas centrado
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

# ---------- util: overlay com alpha ----------
def overlay_rgba(dst_bgr, overlay_rgba, x, y):
    """Aplica overlay RGBA (com alpha) sobre dst_bgr em (x,y). Recorta se necessário."""
    oh, ow = overlay_rgba.shape[:2]
    dh, dw = dst_bgr.shape[:2]
    if x >= dw or y >= dh or x+ow <= 0 or y+oh <= 0:
        return  # totalmente fora
    # ajustes se passar das bordas
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + ow, dw)
    y1 = min(y + oh, dh)
    ox0 = x0 - x
    oy0 = y0 - y
    roi = dst_bgr[y0:y1, x0:x1]
    ov = overlay_rgba[oy0:oy0+(y1-y0), ox0:ox0+(x1-x0)]
    if ov.shape[2] == 4:
        overlay_rgb = ov[:, :, :3].astype(np.float32)
        alpha = (ov[:, :, 3:4].astype(np.float32)) / 255.0
        base = roi.astype(np.float32)
        blended = alpha * overlay_rgb + (1 - alpha) * base
        dst_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)
    else:
        # sem alpha: simples colagem
        dst_bgr[y0:y1, x0:x1] = ov[:, :, :3]

# ---------- prepara tela cheia e logo ----------
SCREEN_W, SCREEN_H = get_screen_resolution()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# tela cheia
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# carrega e pré-processa a logo (mantém aspecto e dimensiona à fração da tela)
logo_rgba = None
if LOGO_PATH:
    tmp = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)  # tenta RGBA
    if tmp is not None:
        # garante 4 canais
        if tmp.shape[2] == 3:
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2BGRA)
        # redimensiona proporcionalmente para ocupar a fração definida da largura da tela
        desired_w = max(1, int(SCREEN_W * LOGO_SCREEN_WIDTH_FRAC))
        h, w = tmp.shape[:2]
        scale = desired_w / w
        new_w = desired_w
        new_h = max(1, int(round(h * scale)))
        logo_rgba = cv2.resize(tmp, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Estado
joints = [90, 90, 90, 90]    # J1..J4
last_base = None
last_j_payload = None
last_joint_send = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # linhas para dividir metade esquerda (seleção de juntas)
    left_w = W // 2
    mid_x = left_w // 2
    mid_y = H // 2

    to_send_base = None
    updated_joint = False
    selected_joint = None

    if results.multi_hand_landmarks and results.multi_handedness:
        # monta pares
        infos = []
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hd.classification[0].label  # 'Left' ou 'Right'
            score = hd.classification[0].score
            infos.append((lm, label, score))
        infos.sort(key=lambda x: x[2], reverse=True)
        infos = infos[:2]

        left_hand = None
        right_hand = None
        for full, label, _ in infos:
            if label == 'Left' and left_hand is None:
                left_hand = full
            elif label == 'Right' and right_hand is None:
                right_hand = full

        # ====== MÃO DIREITA -> LOCOMOÇÃO ======
        if right_hand is not None:
            r_lm = right_hand.landmark
            ud = gesture_thumb_ud(r_lm)
            lr = gesture_thumb_lr(r_lm)
            to_send_base = base_cmd_from_dirs(ud, lr)
            mp_draw.draw_landmarks(frame, right_hand, mp.solutions.hands.HAND_CONNECTIONS)
            txt = "BASE:"
            if ud: txt += f" {ud}"
            elif lr: txt += f" {lr}"
            else: txt += " NEUTRO"
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            to_send_base = NEUTRO_BASE

        # ====== MÃO ESQUERDA -> MANIPULADOR (4 QUADRANTES) ======
        if left_hand is not None:
            mp_draw.draw_landmarks(frame, left_hand, mp.solutions.hands.HAND_CONNECTIONS)
            lmk = left_hand.landmark
            # usa o "centro" como média dos pontos (ou wrist)
            cx = int(np.mean([p.x for p in lmk]) * W)
            cy = int(np.mean([p.y for p in lmk]) * H)

            # Só considera controle se a mão estiver na metade ESQUERDA
            if cx < left_w:
                # quadrantes na metade esquerda:
                # Q1: topo-esq -> J1
                # Q2: topo-dir -> J2
                # Q3: baixo-esq -> J3
                # Q4: baixo-dir -> J4
                if cx < mid_x and cy < mid_y:   selected_joint = 0  # J1
                elif cx >= mid_x and cy < mid_y: selected_joint = 1  # J2
                elif cx < mid_x and cy >= mid_y: selected_joint = 2  # J3
                else:                             selected_joint = 3  # J4

                # gesto joia cima/baixo para ajustar
                ud = gesture_thumb_ud(lmk)
                if ud == "UP":
                    joints[selected_joint] = clamp(joints[selected_joint] + JOINT_STEP, JOINTS_MIN, JOINTS_MAX)
                    updated_joint = True
                elif ud == "DOWN":
                    joints[selected_joint] = clamp(joints[selected_joint] - JOINT_STEP, JOINTS_MIN, JOINTS_MAX)
                    updated_joint = True

            # overlay da seleção
            color_sel = (0, 255, 0)
            cv2.circle(frame, (cx, cy), 8, color_sel, -1)

    else:
        to_send_base = NEUTRO_BASE  # segurança

    # ====== ENVIO BASE ======
    if to_send_base is not None and to_send_base != last_base:
        http_send(to_send_base)
        last_base = to_send_base

    # ====== ENVIO JUNTAS (debounce simples) ======
    now = time.time()*1000
    if updated_joint and (now - last_joint_send >= SEND_DEBOUNCE_MS):
        payload = build_joints_payload(joints)
        if payload != last_j_payload:
            http_send(payload)
            last_j_payload = payload
        last_joint_send = now

    # ====== DESENHO DOS 4 QUADRANTES NA METADE ESQUERDA ======
    # área esquerda
    cv2.line(frame, (left_w, 0), (left_w, H), (255,255,255), 1)
    # grade na metade esquerda
    cv2.line(frame, (mid_x, 0), (mid_x, H), (200,200,200), 1)
    cv2.line(frame, (0, mid_y), (left_w, mid_y), (200,200,200), 1)

    # rótulos J1..J4
    cv2.putText(frame, "J1", (mid_x//2 - 10, mid_y//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
    cv2.putText(frame, "J2", (mid_x + (mid_x//2) - 10, mid_y//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
    cv2.putText(frame, "J3", (mid_x//2 - 10, mid_y + (mid_y//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
    cv2.putText(frame, "J4", (mid_x + (mid_x//2) - 10, mid_y + (mid_y//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

    # valores atuais
    vals = f"Juntas: {int(joints[0])},{int(joints[1])},{int(joints[2])},{int(joints[3])}"
    cv2.putText(frame, vals, (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # destaca quadrante selecionado
    if 'selected_joint' in locals() and selected_joint is not None:
        if selected_joint == 0:
            x0,y0,x1,y1 = 0,0, mid_x, mid_y
        elif selected_joint == 1:
            x0,y0,x1,y1 = mid_x,0, left_w, mid_y
        elif selected_joint == 2:
            x0,y0,x1,y1 = 0, mid_y, mid_x, H
        else:
            x0,y0,x1,y1 = mid_x, mid_y, left_w, H
        cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 6)

    # ====== AJUSTE PARA TELA CHEIA (sem distorcer) ======
    # redimensiona com letterbox para caber na tela toda
    frame_full = resize_letterbox(frame, SCREEN_W, SCREEN_H)

    # ====== LOGO NO CANTO INFERIOR DIREITO ======
    if logo_rgba is not None:
        lh, lw = logo_rgba.shape[:2]
        # posição: canto inferior direito com margem
        x = SCREEN_W - lw - LOGO_MARGIN_PX
        y = SCREEN_H - lh - LOGO_MARGIN_PX
        overlay_rgba(frame_full, logo_rgba, x, y)

    cv2.imshow(WINDOW_NAME, frame_full)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
