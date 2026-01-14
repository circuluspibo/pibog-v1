"""
#모듈:값!
 
 - 모터 제어 
   send >>> #motor:20!  / #motor:-20!
   receive 없음
 - 센서 (온도/습도/가스)
   send >>>  #env:!
   receive >>> 21.0,40.5,50.0   (온도,습도,가스)

def mq9_grade_from_adc(value: int) -> str:
    """
    MQ-9 아날로그 값(0~1023)을
    VG / G / N / B / VB 로 변환
    """
    if value <= 200:
        return "VG"   # Very Good
    elif value <= 350:
        return "G"    # Good
    elif value <= 500:
        return "N"    # Normal
    elif value <= 700:
        return "B"    # Bad
    else:
        return "VB"   # Very Bad
"""
import serial

class UsbUart:
  def __init__(self):
    pass

  def connect(self, devname='/dev/ttyUSB0', baudrate=9600, timeout=1):
    try:
      self.conn = serial.Serial(devname, baudrate, timeout=timeout)
    except serial.SerialException as e:
      raise ConnectionError(f"시리얼 포트 {devname} 열기에 실패했습니다: {e}")

  def write(self, text):
    print(str(text).encode('utf-8'),"cmd")
    if self.conn.is_open:
      return self.conn.write(str(text).encode('utf-8'))
    else:
      raise ConnectionError("시리얼 포트가 열려 있지 않습니다.")

  def read(self):
    if self.conn.is_open:
      response = self.conn.read_all()
      return response.decode('utf-8', errors='ignore') if response else ""
    else:
      raise ConnectionError("시리얼 포트가 열려 있지 않습니다.")

  def close(self):
    if self.conn.is_open:
      self.conn.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

if __name__ == "__main__":
    uu = UsbUart()
    uu.connect()

    while True:
        command = input("command > ")

        if command == "exit":
            break
        elif "motor" in command:
            uu.write(command)
        elif "env" in command:
            uu.write(command)
            res = uu.read()
            print(res)
