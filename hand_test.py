from mandro import HadnControler
import threading
import readline

try:
    hL = HadnControler('/dev/ttyACM0') # L 컨트롤러 L동글 부터 연결
    hR = HadnControler('/dev/ttyACM1') # R 컨트롤러
    print("컨트롤러 초기화 성공")
except Exception as e:
    print(f"컨트롤러 초기화 실패: {e}")
    exit()

def send_command_concurrently_thread(command):
    thread_L = threading.Thread(target=hL.send_motion, args=(command,))
    thread_R = threading.Thread(target=hR.send_motion, args=(command,))

    thread_L.start()
    thread_R.start()

while True:
    # Example usage
    command_name = input("Enter command name (or 'exit' to quit): ")
    if command_name == "exit":
        break

    send_command_concurrently_thread(command_name)
    # hL.send_motion(command_name) #L
    # hR.send_motion(command_name) #R
 
