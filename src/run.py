import subprocess
import sys

# 使用 python -m 运行 src.main
subprocess.run([sys.executable, "-m", "src.train"] + sys.argv[1:])