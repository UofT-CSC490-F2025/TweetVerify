import subprocess
import sys

# 使用 python -m 运行 src.main
subprocess.run([sys.executable, "-m", "src.main"] + sys.argv[1:])