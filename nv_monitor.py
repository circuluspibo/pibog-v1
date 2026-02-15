import os
import threading
import time

class JetsonPowerMonitor:
    def __init__(self, hwmon_path="/sys/class/hwmon/hwmon5", interval=1.0):
        self.hwmon_path = hwmon_path
        self.interval = interval

        self.gpu_power = 0.0
        self.cpu_power = 0.0
        self.sys5v_power = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def _read_value(self, filename):
        try:
            with open(os.path.join(self.hwmon_path, filename), "r") as f:
                return float(f.read().strip())
        except Exception:
            return 0.0

    def _calculate_power(self, curr_file, volt_file):
        curr_mA = self._read_value(curr_file)
        volt_mV = self._read_value(volt_file)

        # W = (mA × mV) / 1,000,000
        return (curr_mA * volt_mV) / 1_000_000

    def _monitor(self):
        while not self._stop_event.is_set():

            # Channel 1 → GPU
            self.gpu_power = self._calculate_power("curr1_input", "in1_input")

            # Channel 2 → CPU + SOC
            self.cpu_power = self._calculate_power("curr2_input", "in2_input")

            # Channel 3 → 5V rail
            self.sys5v_power = self._calculate_power("curr3_input", "in3_input")

            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    # 🔥 추가된 함수
    def get_power(self):
        """Return total SoC power (GPU + CPU/SOC)."""
        return self.gpu_power + self.cpu_power

    def get_all(self):
        return {
            "GPU": self.gpu_power,
            "CPU_SOC": self.cpu_power,
            "SYS_5V": self.sys5v_power,
            "TOTAL_SOC": self.get_power()
        }


# ==========================
# Usage Example
# ==========================
if __name__ == "__main__":
    monitor = JetsonPowerMonitor(interval=1.0)
    monitor.start()

    try:
        print("Monitoring hwmon power consumption... (Ctrl+C to stop)\n")
        while True:
            total = monitor.get_power()
            data = monitor.get_all()

            print(
                f"GPU: {data['GPU']:.2f} W | "
                f"CPU+SOC: {data['CPU_SOC']:.2f} W | "
                f"5V: {data['SYS_5V']:.2f} W | "
                f"TOTAL_SOC: {total:.2f} W"
            )

            time.sleep(1)

    except KeyboardInterrupt:
        monitor.stop()
        print("\nStopped monitoring.")


"""
import subprocess
import re
import threading
import time

class CPUPowerMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.gpu_power = 0.0
        self.cpu_power = 0.0
        self.sys_power = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        
    def _read_power(self):
        """Run tegrastats and extract power readings."""
        process = subprocess.Popen(
            ["tegrastats", "--interval", str(int(self.interval * 1000))],  # tegrastats reports in milliseconds
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        while not self._stop_event.is_set():
            line = process.stdout.readline()
            if not line:
                continue
            
            # Example output line:
            # VDD_GPU_SOC 2416mW/2617mW VDD_CPU_CV 402mW/604mW VIN_SYS_5V0 3628mW/3729mW
            
            # Extracting the relevant power fields using regex
            gpu = re.search(r"VDD_GPU_SOC\s+(\d+)mW", line)
            cpu = re.search(r"VDD_CPU_CV\s+(\d+)mW", line)
            vin = re.search(r"VIN_SYS_5V0\s+(\d+)mW", line)

            if gpu or cpu or vin:
                self.gpu_power = int(gpu.group(1)) / 1000 if gpu else 0
                self.cpu_power = int(cpu.group(1)) / 1000 if cpu else 0
                self.sys_power = int(vin.group(1)) / 1000 if vin else 0

            # Sleep for the desired interval before reading the next output
            time.sleep(self.interval)

    def _monitor(self):
        """Background thread for monitoring power."""
        self._read_power()

    def start(self):
        """Start the background monitoring thread."""
        self._thread.start()

    def stop(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        self._thread.join()

    def get_power(self):
        """Return the latest measured powers."""

        return self.gpu_power + self.cpu_power

    def get_all(self):
        """Return the latest measured powers."""
        return {
            "GPU": self.gpu_power,
            "CPU": self.cpu_power,
            "System(5V)": self.sys_power
        }

# ==========================
# Usage Example
# ==========================
if __name__ == "__main__":
    monitor = CPUPowerMonitor(interval=1.0)  # Monitor every second
    monitor.start()

    try:
        print("Monitoring power consumption... (Ctrl+C to stop)\n")
        while True:
            power_data = monitor.get_power()
            print(f"GPU: {power_data['GPU']:.2f} W   CPU: {power_data['CPU']:.2f} W   SYS(5V): {power_data['System(5V)']:.2f} W")
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("\nStopped monitoring.")
"""
