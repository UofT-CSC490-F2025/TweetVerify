import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "src.train"] + sys.argv[1:])

if __name__ == "__main__":
    main()