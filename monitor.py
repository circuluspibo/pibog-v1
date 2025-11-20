
#sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj

import threading
import time
import os

class CPUPowerMonitor:
    def __init__(self, rapl_path="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", interval=1.0):
        self.rapl_path = rapl_path
        self.interval = interval
        self.power = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        
        if not os.path.exists(self.rapl_path):
            raise FileNotFoundError(f"RAPL interface가 {self.rapl_path}에 없습니다.")

    def _read_energy(self):
        with open(self.rapl_path, "r") as f:
            energy_uj = int(f.read().strip())
        return energy_uj / 1_000_000  # μJ -> J

    def _monitor(self):
        last_energy = self._read_energy()
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            current_energy = self._read_energy()
            self.power = (current_energy - last_energy) / self.interval
            last_energy = current_energy

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def get_power(self):
        """최근 측정된 CPU 전력(W)을 반환"""
        return self.power


# ==========================
# 사용 예제
# ==========================
if __name__ == "__main__":
    monitor = CPUPowerMonitor(interval=1.0)
    monitor.start()
    
    try:
        print("백그라운드 CPU 전력 모니터링 시작 (종료: Ctrl+C)")
        while True:
            print(f"현재 CPU 전력: {monitor.get_power():.2f} W")
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("측정 종료")

