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
