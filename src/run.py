import subprocess
import sys


subprocess.run([sys.executable, "-m", "src.train"] + sys.argv[1:])