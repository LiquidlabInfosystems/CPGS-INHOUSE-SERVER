#!/usr/bin/env python3
import os
import sys
import subprocess

def setup_wifi_script():
    """Setup the WiFi configuration script on Raspberry Pi"""
    try:
        # Create the script directory if it doesn't exist
        script_dir = "/usr/local/bin"
        os.makedirs(script_dir, exist_ok=True)
        
        # Copy the update_wifi.py script to the system directory
        print("Installing WiFi configuration script...")
        subprocess.run(f"sudo cp update_wifi.py {script_dir}/update_wifi", shell=True, check=True)
        
        # Make the script executable
        subprocess.run(f"sudo chmod +x {script_dir}/update_wifi", shell=True, check=True)
        
        # Create a systemd service for automatic connection
        print("Creating systemd service...")
        service_content = """[Unit]
Description=WiFi Configuration Service
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/update_wifi
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        
        # Write the service file
        with open("/tmp/wifi-config.service", "w") as f:
            f.write(service_content)
        
        # Install the service
        subprocess.run("sudo mv /tmp/wifi-config.service /etc/systemd/system/", shell=True, check=True)
        subprocess.run("sudo systemctl daemon-reload", shell=True, check=True)
        subprocess.run("sudo systemctl enable wifi-config.service", shell=True, check=True)
        
        print("\nSetup completed successfully!")
        print("\nTo use the WiFi configuration:")
        print("1. Run: sudo update_wifi <SSID> <PASSWORD>")
        print("2. Reboot: sudo reboot")
        print("\nThe device will automatically connect to the configured network after reboot.")
        
        return True
        
    except Exception as e:
        print(f"Error during setup: {e}")
        return False

if __name__ == "__main__":
    if setup_wifi_script():
        print("\nWiFi configuration setup completed!")
    else:
        print("\nFailed to setup WiFi configuration")
        sys.exit(1) 