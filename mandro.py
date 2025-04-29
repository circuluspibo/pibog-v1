import serial
import time
import math

# 엄지 손가락 위치 조정
def f1_pos(ratio):
  byte_list = [0x00, 0x70]
  combined_value = (byte_list[0] << 8) | byte_list[1]
  result_int = math.floor(combined_value * ratio)
  return [(result_int >> 8) & 0xFF, result_int & 0xFF]

# 엄지 제외 손가락 위치 조정
def f24_pos(ratio):
  byte_list = [0x01, 0x20]
  combined_value = (byte_list[0] << 8) | byte_list[1]
  result_int = math.floor(combined_value * ratio)
  return [(result_int >> 8) & 0xFF, result_int & 0xFF]

def create_command(fingers, action, ratio):
    if action == "fold":
        direction = 0x01
    elif action == "unfold":
        direction = 0x02
    else:
        direction = 0x0

    if fingers[0]:
        position = f1_pos(ratio)
    else:
        position = f24_pos(ratio)

    command = [0xAA, 0x55]                 # Bytes 0, 1: 헤더
    command.extend(fingers)          # Bytes 2-6: 손가락 활성화 상태
    command.extend([0x08, 0xFC, 0x04, 0x30]) # Bytes 7-8 speed, Bytes 9-10: current
    command.extend(position)         # Bytes 11, 12: 계산된 위치값
    command.append(direction)              # Byte 13: 계산된 방향
    command.extend([0,0])                 # Bytes 14: 체크섬 자리
    return {'pos': command, 'time': sum(e==1 for e in fingers) * ratio*0.8}

# --- 손동작 명령어 딕셔너리 생성 ---
# 손가락 상태: [엄지, 검지, 중지, 약지, 새끼] (1=활성, 0=비활성)
# !!! 엄지 / 나머지 4개 손가락 명령어 분리 해야 함

motions = {
    # --- 기본 동작 ---
    "fold_a": [ # all
        create_command([1, 0, 0, 0, 0], "fold",     1),
        create_command([0, 1, 1, 1, 1], "fold",     1),
    ],
    "unfold_a": [ #all
        create_command([0, 1, 1, 1, 1], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "unfold",   1),
    ],
    "fold_ha": [ #half all
        create_command([1, 0, 0, 0, 0], "fold",   0.5),
        create_command([0, 1, 1, 1, 1], "fold",   0.5),
    ],
    "unfold_ha": [ #half all
        create_command([0, 1, 1, 1, 1], "unfold", 0.5),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],

    # --- 조합 동작 ---
    "point": [
        #create_command([0, 1, 0, 0, 0], "unfold",   1),

        create_command([0, 0, 1, 1, 1], "fold",     1),
        create_command([1, 0, 0, 0, 0], "fold",     1),
    ],
    "point-exit": [
        create_command([0, 0, 1, 1, 1], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "unfold",   1),
    ],

    "handshake": [
        #create_command([0, 1, 1, 1, 1], "unfold",   1),
        #create_command([1, 0, 0, 0, 0], "unfold",   1),
        
        create_command([0, 1, 1, 1, 1], "fold",   0.3),
    ],
    "handshake-exit": [
        create_command([0, 1, 1, 1, 1], "unfold", 0.3),
    ],

    "ok": [
        #create_command([0, 0, 1, 1, 1], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "fold",   0.5),
        create_command([0, 1, 0, 0, 0], "fold",   0.9),
    ],
    "ok-exit": [
        create_command([0, 1, 0, 0, 0], "unfold", 0.9),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],

    "thumbup": [
        #create_command([1, 0, 0, 0, 0], "unfold",   1),
        create_command([0, 1, 1, 1, 1], "fold",     1),
    ],
    "thumbup-exit": [
        create_command([0, 1, 1, 1, 1], "unfold",   1),
    ],

    "victory": [
        #create_command([0, 1, 1, 0, 0], "unfold",   1), 
        create_command([1, 0, 0, 0, 0], "fold",     1),
        create_command([0, 0, 0, 1, 1], "fold",     1),
    ],
    "victory-exit": [
        create_command([0, 0, 0, 1, 1], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "unfold",   1),
    ],

    "rock": [
        #create_command([0, 1, 0, 0, 1], "unfold",   1), 
        #create_command([1, 0, 0, 0, 0], "unfold",   1), 
        create_command([0, 0, 1, 1, 0], "fold",     1),
    ],
    "rock-exit": [
        create_command([0, 0, 1, 1, 0], "unfold",   1),
    ],

    "promise": [
        #create_command([0, 0, 0, 0, 1], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "fold",     1),
        create_command([0, 1, 1, 1, 0], "fold",     1),
    ],
    "promise-exit": [
        create_command([0, 1, 1, 1, 0], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "unfold",   1),
    ],

    "grab": [
        #create_command([0, 1, 1, 1, 1], "unfold",   1),
        #create_command([1, 0, 0, 0, 0], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "fold",   0.5),
        create_command([0, 1, 1, 1, 1], "fold",   0.5),
    ],
    "grab-exit": [
        create_command([0, 1, 1, 1, 1], "unfold", 0.5),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],
}
#print([ name for name in motions])

class HadnControler:
    def __init__(self, port):
        self.port = port
        self.ser = serial.Serial(port=port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=3)
        #self.send_motor([0xAA, 0x55, 0x01, 0x01, 0x01, 0x01, 0x01,  0x08, 0xFC,  0x04, 0x30,  0x00, 0x00, 0x00, 0x0])

    def send_motor(self, data):
        cmd_data = data.copy()
        chksum = 0
        for value in cmd_data[2:14]:
            chksum ^= value
        cmd_data[14] = chksum
        data_bytes = bytes(cmd_data)
        try:
            self.ser.write(data_bytes)
        except Exception as e:
            print(f"[Failed to send command: {e}")
    
    def send_motion(self, motion_name):
        if motion_name not in motions:
            print(f"Invalid command name: {motion_name}")
            return
        
        items = motions[motion_name]
        for item in items:
            #print([hex(i) for i in item["pos"]],item["time"])
            self.send_motor(item["pos"])
            time.sleep(item["time"])
        
    def close(self):
        self.ser.close()
