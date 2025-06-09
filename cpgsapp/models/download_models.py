import os
import urllib.request
import gzip
import shutil
import ssl
import sys

def download_file(url, filename):
    """Download a file with proper error handling and SSL verification."""
    print(f"Downloading {filename}...")
    try:
        # Create SSL context that ignores certificate verification
        ssl_context = ssl._create_unverified_context()
        
        # Download with progress reporting
        with urllib.request.urlopen(url, context=ssl_context) as response, open(filename, 'wb') as out_file:
            file_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            downloaded = 0
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                    
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # Calculate and display progress
                if file_size > 0:
                    percent = int(downloaded * 100 / file_size)
                    sys.stdout.write(f"\rProgress: {percent}%")
                    sys.stdout.flush()
            
            print(f"\nDownloaded {filename}")
            return True
            
    except Exception as e:
        print(f"\nError downloading {filename}: {str(e)}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(models_dir, exist_ok=True)

    # Model files and their URLs
    model_urls = {
        'yolov3-tiny.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
        'yolov3-tiny.weights': 'https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.weights',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }

    # Download each file if it doesn't exist
    success = True
    for filename, url in model_urls.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            if not download_file(url, filepath):
                success = False
                print(f"Failed to download {filename}")
        else:
            print(f"{filename} already exists, skipping download")

    if success:
        print("\nAll model files downloaded successfully!")
    else:
        print("\nSome files failed to download. Please check the errors above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 