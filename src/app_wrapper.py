import subprocess
import sys

p1 = subprocess.Popen([sys.executable, "-m", "src.app"])
p2 = subprocess.Popen([sys.executable, "-m", "src.auth_app"])

print(f"Started: {p1.pid}, {p2.pid}")
