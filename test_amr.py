import requests
import time

# 로봇의 기본 주소 (localhost를 실제 로봇 IP로 변경 가능)
BASE_URL = "http://10.42.0.87"

def send_command(endpoint, data):
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"성공: {endpoint} | 데이터: {data}")
        else:
            print(f"오류: {response.status_code} | {response.text}")
    except Exception as e:
        print(f"연결 실패: {e}")

# 1. 앞으로 50cm 이동
# direction 1: forward, speed: 0.3 m/s (기본값 예시)
move_forward = {"distance": 100, "direction": 1, "speed": 0.5}
send_command("/cmd/move", move_forward)
time.sleep(3)  # 이동 시간 대기

# 2. 뒤로 50cm 이동
# direction 0: backward
#move_backward = {"distance": 100, "direction": 0, "speed": 0.5}
#send_command("/cmd/move", move_backward)
#time.sleep(3)


# 3. 왼쪽으로 회전 (90도 예시)
# direction 1: left, angle: 90, speed: 0.5 rad/s
turn_left = {"direction": 1, "angle": 90, "speed": 0.5}
send_command("/cmd/turn", turn_left)
time.sleep(2)

# 4. 회전 후 앞으로 50cm 이동
send_command("/cmd/move", move_forward)
time.sleep(3)

# 5. 오른쪽으로 회전 (90도 예시)
# direction 0: right
turn_right = {"direction": 0, "angle": 180, "speed": 0.5}
send_command("/cmd/turn", turn_right)
time.sleep(2)

# 6. 다시 앞으로 50cm 이동
turn_right = {"direction": 0, "angle": 90, "speed": 0.5}
send_command("/cmd/move", move_forward)

# 정지 명령 (필요 시)
stop_move = {"distance": 0, "direction": 1, "speed": 0}
send_command("/cmd/move", stop_move)
