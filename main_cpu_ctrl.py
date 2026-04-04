# PC2
# sudo nano /etc/sysctl.conf
# net.ipv4.ip_forward=1
# sudo sysctl -p
# sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
#sudo apt install iptables-persistent
#sudo netfilter-persistent save

# Intel / pc3
# sudo nano /etc/netplan/01-netcfg.yaml
"""
network:
  version: 2
  renderer: networkd
  wifis:
    wlan0:
      dhcp4: yes
      routes:
        - to: 192.168.123.0/24
          via: 192.168.21.19
"""
# sudo netplan apply

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import numpy as np
from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, VUI_COLOR, SPORT_CMD
import logging
import matplotlib.colors
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import requests

#optimum-cli export openvino --weight-format int4 --task text-generation-with-past --model growdle/HyperCLOVAX-SEED-Text-Instruct-1.5B ./CLOVAX-1.5B-ov-int4
#kakaocorp/kanana-1.5-2.1b-instruct-2505
#https://github.com/Unitree-Go2-Robot/go2_robot


def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

_IP = "192.168.21.19"

# "192.168.12.117" - g1 basic
# "192.168.12.112" - g1 plus local
# "192.168.12.128" - g1 basic pohang
# "192.168.21.19" - g1 plus

print(_IP)

# Enable logging for debugging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_PORT = int(open("port.txt", 'r').read())


conn = None
lastCmd = {} 

G1_ARM = {
  "clamp": 17, 
  "highFive": 18, 
  "shakeHands_1": 27,
  "makeHeartBothHands": 20, 
  "makeHeartSingleHands": 21,
  "blowKiss": 12, 
  "hug": 19,
  "hightWave": 26, 
  "lowWave" : 25,
  "ultramanRay" : 24, 
  "bothHandsUp" : 15,
  "singleHandsUp" : 23,
  "Refuse" : 22, 
  "Release_Arm" : 99,
}

G1_STATE = {
  "ZeroTorque" : 0,
  "Damp" : 1,
  "Preparation": 4,
  "Seating": 3,       
  "Walk_G1": 500,
  "Walk2_G1" : 501,
  "Run_G1" : 801,
  "Squat_G1" : 706,  
  "SquatUp_G1" : 706,
  "LieUp_G1" : 702,
}

G1_BALANCE = {
  "Stand_G1" : 0,
  "Step_G1" : 1 
}

app = FastAPI()

app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)

state = { "charge" : 0, "temp" : 0, "voltage" : 0 }

@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      

@app.get("/connect")
async def connect2():
  global conn
  #global audio_hub
  conn =  UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.123.161") #Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.0.101")
  await conn.connect()
  print(1)
  #audio_hub = WebRTCAudioHub(conn, logger)
  #await audio_hub.set_play_mode('no_cycle')
  #conn.video.switchVideoChannel(True)
  #conn.video.add_track_callback(recv_camera_stream)
  #print(3)
  def lowstate_callback(message):
    #print(message)
    msg = message['data']      
    state["charge"] = msg['bms_state']['soc']
    state["temp"] = msg['temperature_ntc1']
    state["voltage"] = msg['power_v']

  conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LOW_STATE'], lowstate_callback)

  return { "result" : True, "data" : True }     

@app.get("/sport")
async def sport(cmd : str, x=0.0, y=0.0, z=0.0, data=None):
  global conn
  out = 0
  print(cmd, f'x:{x}, y:{y}, z:{z}')
  if conn is None:
    print('Disconnected', cmd)
  elif cmd == 'Move':
    out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
          "api_id": SPORT_CMD[cmd],
          "parameter": {"x": float(x), "y": float(y), "z": float(z)}
        }
    )
  elif data != None: # SPORT_CMD[cmd] != None and
    out = await conn.datachannel.pub_sub.publish_request_new(
      RTC_TOPIC["SPORT_MOD"], {
          "api_id": SPORT_CMD[cmd],
          "parameter": { "data" : float(data) } # if possible
      }
    )
  else:
    if lastCmd.get(cmd, False):
      lastCmd[cmd] = True
      out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
            "api_id": SPORT_CMD[cmd],
            "parameter": { "data" : True } # if possible
        }
      )       
    else:
      lastCmd[cmd] = False
      out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
            "api_id": SPORT_CMD[cmd],
            "parameter": { "data" : False } # if possible
        }
      )


  print("response", out)
            
  return { "result" : True, "data" : out }      

def toFloat(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

@app.get("/manual")
async def manual(cmd : str, data : str):
  out = await conn.datachannel.pub_sub.publish_request_new(
    RTC_TOPIC["SPORT_MOD"], {
        "api_id": int(cmd),
        "parameter": { "data" : toFloat(data) } # if possible
    }
  )

  print("response", out)
            
  return { "result" : True, "data" : out }      

@app.get("/arm")
async def arm(cmd = "clamp"):
  global conn
  global G1_ARM

  await conn.datachannel.pub_sub.publish_request_new(
    "rt/api/arm/request", {
        "api_id": 7106,
        "parameter" : { "data" : G1_ARM[cmd] }
    }
  )

  return { "result" : True, "data" : True }      

@app.get("/walkG1")
async def walkG1(lx = 0, ly = 0, rx = 0, ry = 0):
  print("walking",f"L : {lx} {ly} | R : {rx} {ry}")
  global conn

  conn.datachannel.pub_sub.publish_without_callback(
     "rt/wirelesscontroller", {
        "lx": float(lx), "ly": float(ly), "rx": float(rx), "ry": float(ry) 
     }
  )
  """
  await conn.datachannel.pub_sub.publish("rt/wirelesscontroller", { 
     "lx": int(lx), "ly": int(ly), "rx": int(rx), "ry": int(ry) 
  })
  """
  
  return { "result" : True, "data" : True }     

@app.get("/stateG1")
async def stateG1(cmd="Walk_G1"):
  global conn
  global G1_STATE

  await conn.datachannel.pub_sub.publish_request_new(
    "rt/api/sport/request", {
        "api_id": 7101,
        "parameter" : { "data" : G1_STATE[cmd] }
    }
  )

  return { "result" : True, "data" : True }      

@app.get("/balanceG1")
async def balanceG1(cmd="Stand_G1"):
  global conn
  global G1_BALANCE

  await conn.datachannel.pub_sub.publish_request_new(
    "rt/api/sport/request", {
        "api_id": 7102,
        "parameter" : { "data" : G1_BALANCE[cmd] }
    }
  )

  return { "result" : True, "data" : True }      

@app.get("/color")
async def color(value = 'purple', warn = 0):
  global conn
  global lastColor 

  if lastColor == value:
    return

  if conn is None:
    print('brightness', value)
  else:  
    lastColor = value
    await conn.datachannel.pub_sub.publish_request_new(
      RTC_TOPIC["VUI"], 
      {
        "api_id": 1007,
        "parameter": 
        {
            "color": value,
            "time": 5,
            "flash_cycle": 1000  # Flash every second
        }
      }
    )

  return { "result" : True, "data" : True }  

@app.get("/brightness")
async def brightness(value = 10):
  global conn

  if conn is None:
    print('brightness', value)
  else:
    await conn.datachannel.pub_sub.publish_request_new(
      RTC_TOPIC["VUI"], 
      {
          "api_id": 1005,
          "parameter": {"brightness": int(value)}
      }
    )

  return { "result" : True, "data" : True } 

@app.get("/mode")
async def mode(value = 'normal'):
  global conn
  if conn is None:
    print('mode', value)
  else:  
    conn.datachannel.pub_sub.publish_without_callback(
      RTC_TOPIC["MOTION_SWITCHER"], 
      {
          "api_id": 1002,
          "parameter": {"name": value}
      }
    )

  return { "result" : True, "data" : True }  

@app.get("/volume")
async def volume(value = 10):
  global conn
  if conn is None:
    print('volume', value)
  else:
    conn.datachannel.pub_sub.publish_without_callback(
      RTC_TOPIC["VUI"], 
      {
          "api_id": 1003,
          "parameter": {"volume": int(value)}
      }
    )

  return { "result" : True, "data" : True }  


@app.get("/color")
async def color(value : str = 'red'):
  print(value)
  colors = matplotlib.colors.to_rgb(value)
  arr = (np.array(colors) * 255).astype(int)
  print("color", arr)
  response = requests.get(f"http://{_IP}:59521/led?r={arr[0]}&g={arr[1]}&b={arr[2]}")
  return { "result" : True }

print("TEST","2502010900")
