import socket
import cv2
import numpy as np
import subprocess

# Configurações
WIDTH = 1080  # Ajustar conforme resolução do dispositivo
HEIGHT = 1920
PORT = 5000

# Iniciar servidor TCP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('10.1.1.148', PORT))
server_socket.listen(1)
print("Aguardando conexão Android...")

conn, addr = server_socket.accept()
print(f"Conectado: {addr}")

# Configurar FFmpeg para decodificar H.264
ffmpeg = subprocess.Popen(
    ['ffmpeg', '-i', '-', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)

while True:
    # Ler tamanho do frame (4 bytes)
    size_data = conn.recv(4)
    if not size_data:
        break
    size = int.from_bytes(size_data, byteorder='big')
    
    # Ler dados do frame
    frame_data = b''
    while len(frame_data) < size:
        packet = conn.recv(size - len(frame_data))
        if not packet:
            break
        frame_data += packet
    
    # Enviar para FFmpeg decodificar
    ffmpeg.stdin.write(frame_data)
    ffmpeg.stdin.flush()
    
    # Ler frame decodificado
    raw_frame = ffmpeg.stdout.read(WIDTH * HEIGHT * 3)
    if not raw_frame:
        break
    
    # Exibir com OpenCV
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    cv2.imshow('Android Screen', frame)
    if cv2.waitKey(1) == ord('q'):
        break

ffmpeg.kill()
cv2.destroyAllWindows()
conn.close()