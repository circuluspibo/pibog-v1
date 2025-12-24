import serial
import time
import math

# 엄지 손가락 위치 조정
def f1_pos(ratio):
  byte_list = [0x00, 0x80]
  combined_value = (byte_list[0] << 8) | byte_list[1]
  result_int = math.floor(combined_value * ratio)
  return [(result_int >> 8) & 0xFF, result_int & 0xFF]

# 엄지 제외 손가락 위치 조정
def f24_pos(ratio):
  byte_list = [0x01, 0x50]
  combined_value = (byte_list[0] << 8) | byte_list[1]
  result_int = math.floor(combined_value * ratio)
  return [(result_int >> 8) & 0xFF, result_int & 0xFF]


def create_command(fingers, action, ratio):
    if fingers[0]:
        position = f1_pos(ratio)
    else:
        position = f24_pos(ratio)

    direction = 0x1

    command = [0xFF]                 # Bytes 0 : left(0xFD), right(0xFE), both(0xFF)
    command.extend(fingers)          # Bytes 1-5: 손가락 활성화 상태
    command.extend([0x05, 0xFF, 0x04, 0x30]) # Bytes 6-7 speed, Bytes 8-9: current
    command.extend(position)         # Bytes 10-11: 위치값 (-5 ~ 400)
    command.append(direction)              # Byte 12: 방향 (forward)
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
        create_command([0, 1, 1, 1, 1], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
    ],
    "fold_ha": [ #half all
        create_command([1, 0, 0, 0, 0], "fold",   0.5),
        create_command([0, 1, 1, 1, 1], "fold",   0.5),
    ],
    "unfold_ha": [ #half all
        create_command([0, 1, 1, 1, 1], "unfold", 0.5),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],

    # --- 조합 동작 --o-
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
        create_command([0, 0, 0, 1, 1], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
    ],

    "rock": [
        #create_command([0, 1, 0, 0, 1], "unfold",   1),
        #create_command([1, 0, 0, 0, 0], "unfold",   1),
        create_command([0, 0, 1, 1, 0], "fold",     1),
    ],
    "rock-exit": [
        create_command([0, 0, 1, 1, 0], "unfold",   0.01),
    ],

    "promise": [
        #create_command([0, 0, 0, 0, 1], "unfold",   1),
        create_command([1, 0, 0, 0, 0], "fold",     1),
        create_command([0, 1, 1, 1, 0], "fold",     1),
    ],
    "promise-exit": [
        create_command([0, 1, 1, 1, 0], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
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

releases = {
    "fold_a": [ #all
        create_command([0, 1, 1, 1, 1], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
    ],

    "fold_ha": [ #half all
        create_command([0, 1, 1, 1, 1], "unfold", 0.5),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],


    "point": [
        create_command([0, 0, 1, 1, 1], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
    ],

    "handshake": [
        create_command([0, 1, 1, 1, 1], "unfold", 0.7),
    ],


    "ok": [
        create_command([0, 1, 0, 0, 0], "unfold", 0.1),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],


    "thumbup": [
        create_command([0, 1, 1, 1, 1], "unfold",   0.01),
    ],


    "victory": [
        create_command([0, 0, 0, 1, 1], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
    ],


    "rock": [
        create_command([0, 0, 1, 1, 0], "unfold",   0.01),
    ],

    "promise": [
        create_command([0, 1, 1, 1, 0], "unfold",   0.01),
        create_command([1, 0, 0, 0, 0], "unfold",   0.01),
    ],

    "grab": [
        create_command([0, 1, 1, 1, 1], "unfold", 0.5),
        create_command([1, 0, 0, 0, 0], "unfold", 0.5),
    ],
}

#print([ name for name in motions])

class HandControler:
    def __init__(self, port):
        self.cmd = 'fold_a'
        self.port = port
        self.ser = serial.Serial(port=port, baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=3)
        #self.send_motor([0xAA, 0x55, 0x01, 0x01, 0x01, 0x01, 0x01,  0x08, 0xFC,  0x04, 0x30,  0x00, 0x00, 0x00, 0x0])
        #self.send_motor([0xFF, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03])

    def send_motor(self, data, selector='both'):
        cmd_data = data.copy()

        if selector == 'left':
            cmd_data[0] = 0xFD
        elif selector == 'right':
            cmd_data[0] = 0xFE
        else:
            cmd_data[0] = 0xFF

        print('[send_motor]:', [hex(i) for i in cmd_data], len(cmd_data))
        data_bytes = bytes(cmd_data) + b'\n'  # <-- LF 추가!
        try:
            self.ser.reset_input_buffer()   # 버퍼 클리어 추가
            self.ser.reset_output_buffer()
            self.ser.write(data_bytes)
            self.ser.flush()
            time.sleep(0.1)
            if self.ser.in_waiting:
                print(str(self.ser.readline().decode()))
        except Exception as e:
            print(f"[Failed to send command: {e}")

    def send_motion(self, motion_name, selector='both'):
        self.cmd = motion_name
        if motion_name not in motions:
            print(f"Invalid command name: {motion_name}")
            return

        items = motions[motion_name]
        for item in items:
            self.send_motor(item["pos"], selector)
            time.sleep(item["time"])

    def send_release(self, release_name=None, selector='both'):
        #print('release', release_name)
        if release_name is None or release_name == "":
            release_name = self.cmd

        items = releases[release_name]
        for item in items:
            #print([hex(i) for i in item["pos"]],item["time"])
            self.send_motor(item["pos"], selector)
            time.sleep(item["time"])

    def close(self):
        self.ser.close()
