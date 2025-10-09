import asyncio
import time
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
import asyncio

# G1 로봇의 IP 주소
ROBOT_IP = "192.168.0.101"

# --- 공식 API 토픽 정보 ---
SPORT_REQUEST_TOPIC = "rt/wirelesscontroller"
ARM_REQUEST_TOPIC = "rt/api/arm/request"
SPORT_REQUEST_TOPIC ="rt/api/sport/request"
# VUI_REQUEST_TOPIC = "rt/api/vui/request" # 예시: LED 제어

# --- API ID (가장 중요한 값, 실제 값으로 변경 필요!) ---
 
G1_STATE_ID = 7101 
G1_BALANCE_ID = 7102
ARM_API_ID = 7106   

LOW_STATE = "rt/lf/lowstate"
LF_SPORT_MOD_STATE =  "rt/lf/bmsstate" 
G1_ARM_ACTION_STATE = "rt/arm/action/state"
BMS_STATE = "rt/lf/bmsstate"
MAIN_BOARD_STATE = "rt/lf/mainboardstate"
LF_SPORT_MOD_STATE = "rt/lf/sportmodestate"

"""
  e.subscribe(RTC_TOPIC.LOW_STATE),
                                e.subscribe(RTC_TOPIC.LF_SPORT_MOD_STATE),
                                e.subscribe(RTC_TOPIC.SLAM_QT_NOTICE),
                                e.subscribe(RTC_TOPIC.SLAM_PC_TO_IMAGE_LOCAL),
                                e.subscribe(RTC_TOPIC.SLAM_RELOCATION_ODOMETRY),
                                e.subscribe(RTC_TOPIC.SLAM_QUERY_ALL_NODE),
                                e.subscribe(RTC_TOPIC.SLAM_QUERY_ALL_EDGE),
MULTIPLE_STATE
mode: e.data.mode || e.data.fsm_id,
            g1_task_id: e.data.task_id,
            gaitType: e.data.gait_type,
            dance: e.data.progress,
            continuousGait: e.data.error_code >> 0 & 1,
            bodyHeight: e.data.body_height,
            footRaiseHeight: e.data.foot_raise_height,
            speedLevel: e.data.progress,
            errorCode: e.data.error_code

"""

# VUI_API_ID = 1007  # 사용자 제공 예시의 VUI API ID

class G1WebRTCController:
    """G1 WebRTC 컨트롤러 (`publish_request_new` 방식 사용)"""
    def __init__(self, ip):
        self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
        self.state0 = False
        self.state1 = False
        self.state2 = False
        self.state3 = False
        self.state4 = False
        self.state5 = False        
        self.is_moving = False

        self.animations = {
            '0': {"name": "clamp", "id": 17}, 
            '1': {"name": "highFive", "id": 18}, 
            '2': {"name": "shakeHands_1", "id": 27},
            '3x': {"name": "X makeHeartBothHands", "id": 20}, 
            '4x': {"name": "X makeHeartSingleHands", "id": 21},
            '3': {"name": "blowKiss", "id": 12}, 
            '4': {"name": "hug", "id": 19},
            '5': {"name": "hightWave", "id": 26}, 
            '6': {"name": "lowWave", "id": 25},
            '7': {"name": "ultramanRay", "id": 24}, 
            '8': {"name": "bothHandsUp", "id": 15},
            '9': {"name": "singleHandsUp", "id": 23},
            '-': {"name": "Refuse", "id": 22}, 
            '=': {"name": "Release Arm", "id": 99},
        }

        self.movement_commands = {
            'a': {"name": "앞으로 이동",  "lx": 2, "ly": 0, "rx": 2, "ry": 0},
            'd': {"name": "뒤로 이동",  "lx": -0.5, "ly": 0, "rx": -0.5, "ry": 0},
            'w': {"name": "앞으로 전진", "lx": 0, "ly": 0.5, "rx": 0, "ry": 0.5},
            'x': {"name": "뒤로 후진", "lx": 0, "ly": -2, "rx": 0, "ry": -2},
            's': {"name": "정지", "lx": 0, "ly": 0, "rx": 0, "ry": 0},
        }

        self.state_commands = {
            'u': {"name": "Damp", "id" : 1},
            'i': {"name": "ZeroTorque", "id" : 0},
            'o': {"name": "Preparation", "id" : 4},
            'p': {"name": "Seating", "id" : 3},       
            'j': {"name": "Walk_G1", "id" : 500},
            'k': {"name": "Walk2_G1", "id" : 501},
            'l': {"name": "Run_G1", "id" : 801},
            'b': {"name": "Squat_G1", "id" : 706},  
            'n': {"name": "SquatUp_G1", "id" : 706},
            'm': {"name": "LieUp_G1", "id" : 702},                            
        }

        self.balance_commands = {
            'z': {"name": "Step_G1", "id" : 1},
            'c': {"name": "Stand_G1", "id" : 0}
        }        

        """

  modeMap[0] = SPORT_STATE.ZeroTorque;
  modeMap[1] = SPORT_STATE.Damp;
  modeMap[2] = SPORT_STATE.Squat_G1;
  modeMap[3] = SPORT_STATE.Seating;
  modeMap[4] = SPORT_STATE.Preparation;
  modeMap[500] = SPORT_STATE.Walk_G1;
  modeMap[501] = SPORT_STATE.Walk2_G1;
  modeMap[706] = SPORT_STATE.Squat_G1;
  modeMap[801] = SPORT_STATE.Run_G1;

          "G1State": {
              "Damp": 1,
              "ZeroTorque": 0,
              "Preparation": 4,
              "Seating": 3,
              "Walk_G1": 500,
              "Walk2_G1": 501,
              "Run_G1": 801,
              "Squat_G1": 706,
              "SquatUp_G1": 706,
              "LieUp_G1": 702
          },
          "G1BalanceState": {
            "Step_G1": 1,
            "Stand_G1": 0
          }
        }
LOW_STATE = "rt/lf/lowstate"
LF_SPORT_MOD_STATE =  "rt/lf/bmsstate" 
G1_ARM_ACTION_STATE = "rt/arm/action/state"
BMS_STATE = "rt/lf/bmsstate"
MAIN_BOARD_STATE = "rt/lf/mainboardstate"
LF_SPORT_MOD_STATE = "rt/lf/sportmodestate"

        """

    async def connect(self):
        print(f"G1 로봇 ({self.conn.ip})에 연결을 시도합니다...")
        # 라이브러리의 connect 메소드가 pub_sub 객체를 초기화한다고 가정
        await self.conn.connect()
        # 데이터 채널이 pub_sub 객체를 가지고 있는지 확인
        if not hasattr(self.conn.datachannel, 'pub_sub'):
             raise AttributeError("사용 중인 라이브러리에 'pub_sub' 객체가 없습니다. 라이브러리 버전을 확인하세요.")
        print("✅ G1 로봇에 성공적으로 연결되었습니다.")

        def s0_cb(message):
            if self.state0 is False:
                self.state0 = True
                print("LOW_STATE", message)
                self.print_instructions()

        def s1_cb(message):
            if self.state1 is False:
                self.state1 = True
                print("LF_SPORT_MOD_STATE", message)    

        def s2_cb(message):
            if self.state2 is False:
                self.state2 = True
                print("G1_ARM_ACTION_STATE", message)    

        def s3_cb(message):
            if self.state3 is False:
                self.state3 = True
                print("BMS_STATE", message)

        def s4_cb(message):
            if self.state4 is False:
                self.state4 = True
                print("MAIN_BOARD_STATE", message)    

        def s5_cb(message):
            if self.state5 is False:
                self.state5 = True
                print("LF_SPORT_MOD_STATE", message)                                             
        
        self.conn.datachannel.pub_sub.subscribe(LOW_STATE, s0_cb)
        self.conn.datachannel.pub_sub.subscribe(LF_SPORT_MOD_STATE, s1_cb)
        self.conn.datachannel.pub_sub.subscribe(G1_ARM_ACTION_STATE, s2_cb)
        self.conn.datachannel.pub_sub.subscribe(BMS_STATE, s3_cb)
        self.conn.datachannel.pub_sub.subscribe(MAIN_BOARD_STATE, s4_cb)
        #time.sleep(3)
        self.print_instructions()

    def print_instructions(self):
        print("\n--- 🤖 G1 조종 방법 (input 명령어 입력 방식) 🤖 ---")
        print("\n[🚶‍♂️ 이동 명령어]")
        print(" w: 앞으로, s: 뒤로, a: 좌회전, d: 우회전")
        print(" wa: 앞+좌회전, wd: 앞+우회전, sa: 뒤+좌회전, sd: 뒤+우회전")
        print(" stop: 모든 동작 정지")
        print("\n[🤸‍♂️ 상반신 동작]")
        for key, anim in self.animations.items():
            print(f" {key}: {anim['name']}")
        print("\n[🚪 종료 및 기타]")
        for key, anim in self.state_commands.items():
            print(f" {key}: {anim['name']}")
        for key, anim in self.balance_commands.items():
            print(f" {key}: {anim['name']}")

        print(" q 또는 quit: 프로그램 종료")
        print(" help: 도움말 다시 보기")
        print("---------------------------------------------------")
        print("💡 명령어를 입력하고 Enter를 누르세요!")

    async def close(self):
        if self.conn.isConnected:
            await self.send_stop_command()
            await self.conn.close()
            print("🔌 연결이 종료되었습니다.")

    async def send_request(self, topic, api_id, params):
        """publish_request_new를 사용하여 API 요청을 보내는 범용 함수"""
        payload = {"api_id": api_id, "parameter": params}
        print(payload)
        await self.conn.datachannel.pub_sub.publish_request_new(topic, payload)

    def send_stop_command(self):
        """로봇의 움직임을 즉시 정지"""
        params = { "lx": 0, "ly": 0, "rx": 0, "ry": 0 }
        self.conn.datachannel.pub_sub.publish_without_callback(SPORT_REQUEST_TOPIC, params)
        #asyncio.create_task(self.conn.datachannel.pub_sub.publish(SPORT_REQUEST_TOPIC, params))
        print("🛑 모든 동작 정지")
        self.is_moving = False

    def send_movement_command(self, lx, ly, rx, ry):
        """이동 명령 전송"""
        params = { "lx": lx, "ly": ly, "rx": rx, "ry": ry }
        print(params)
        self.conn.datachannel.pub_sub.publish_without_callback(SPORT_REQUEST_TOPIC, params)        print(f"🚶‍♂️(lx: {lx},rx: {rx})")
        self.is_moving = True

    async def send_arm_animation_command(self, anim_id, anim_name):
        """상반신 동작 명령 전송"""
        print(f"💃 동작 실행: {anim_name} (ID: {anim_id})")
        params = {"data": anim_id}
        await self.send_request(ARM_REQUEST_TOPIC, ARM_API_ID, params)

    async def send_state_command(self, anim_id, anim_name):
        """상반신 동작 명령 전송"""
        print(f"💃 상태 실행: {anim_name} (ID: {anim_id})")
        params = {"data": anim_id}
        await self.send_request(SPORT_REQUEST_TOPIC, G1_STATE_ID, params)       

    async def send_balance_command(self, anim_id, anim_name):
        """상반신 동작 명령 전송"""
        print(f"💃 균형 실행: {anim_name} (ID: {anim_id})")
        params = {"data": anim_id}
        await self.send_request(SPORT_REQUEST_TOPIC, G1_BALANCE_ID, params)       


    async def process_command(self, command):
        """입력받은 명령어를 처리"""
        command = command.strip().lower()
        
        if not command:
            return True
        
        # 종료 명령
        if command in ['q', 'quit', 'exit']:
            print("프로그램을 종료합니다...")
            return False
        
        # 도움말
        elif command == 'help':
            self.print_instructions()
        
        # 정지 명령
        elif command == 'stop':
            self.send_stop_command()
        
        # 이동 명령
        elif command in self.movement_commands:
            move_info = self.movement_commands[command]
            self.send_movement_command(
                move_info["lx"], 
                move_info["ly"], 
                move_info["rx"],
                move_info["ry"]
            )
        
        # 상반신 동작 명령
        elif command in self.animations:
            anim_info = self.animations[command]
            await self.send_arm_animation_command(anim_info['id'], anim_info['name'])

        elif command in self.state_commands:
            anim_info = self.state_commands[command]
            await self.send_state_command(anim_info['id'], anim_info['name'])   


        elif command in self.balance_commands:
            anim_info = self.balance_commands[command]
            await self.send_balance_command(anim_info['id'], anim_info['name'])        

        # 알 수 없는 명령
        else:
            print(f"❌ 알 수 없는 명령: '{command}'")
            print("💡 'help'를 입력하면 사용 가능한 명령어를 볼 수 있습니다.")
        
        return True

    async def control_loop(self):
        """사용자 입력을 받아 로봇을 제어하는 메인 루프"""
        print("\n🎮 제어 시작! 명령어를 입력하세요:")
        
        while True:
            try:
                # 비동기적으로 사용자 입력 받기
                command = await asyncio.get_event_loop().run_in_executor(
                    None, input, "> "
                )
                
                # 명령어 처리
                should_continue = await self.process_command(command)
                if not should_continue:
                    break
                    
            except KeyboardInterrupt:
                print("\n\nCtrl+C가 감지되었습니다. 프로그램을 종료합니다...")
                break
            except Exception as e:
                print(f"❌ 명령 처리 중 오류: {e}")
                continue


async def main():
    controller = G1WebRTCController(ip=ROBOT_IP)
    try:
        await controller.connect()
        await controller.control_loop()
    except Exception as e:
        print(f"🔥 오류 발생: {e}")
    finally:
        await controller.close()

if __name__ == "__main__":
    try: 
        asyncio.run(main())
    except KeyboardInterrupt: 
        print("\n사용자에 의해 프로그램이 중단되었습니다.")