import requests
import threading
import platform

import subprocess


def get_ssid():
    if platform.system() == "Windows":
        command = "netsh wlan show interfaces"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        for line in result.stdout.splitlines():
            if "SSID" in line:
                return line.split(":")[1].strip()
    elif platform.system() == "Linux":
        command = "iwgetid -r"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return result.stdout.strip()
    else:
        return "Sistema operacional não suportado"

is_sending = False

def fetch_url(url):
    global is_sending
    is_sending = True
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")
    is_sending = False

def start_thread(url):
    global  is_sending
    if(not is_sending):
        thread = threading.Thread(target=fetch_url, args=(url,))
        thread.start()
        return thread
