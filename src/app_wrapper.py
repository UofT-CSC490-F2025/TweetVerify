import subprocess
import sys
from src.utils.get_from_s3 import download_dataset,download_model

def main():
    download_dataset()
    download_model()
    
    p1 = subprocess.Popen([sys.executable, "-m", "src.apps.app"])
    p2 = subprocess.Popen([sys.executable, "-m", "src.apps.auth_app"])
    
    print(f"Started: {p1.pid}, {p2.pid}")

if __name__ == "__main__":
    main()
