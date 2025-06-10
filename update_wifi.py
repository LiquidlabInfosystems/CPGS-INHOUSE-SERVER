#!/usr/bin/env python3
import subprocess
import sys
import time

def update_wifi_config(ssid, password):
    """Update WiFi configuration on Raspberry Pi"""
    try:
        # First, check if the network exists
        print(f"Scanning for WiFi network: {ssid}")
        subprocess.run("sudo nmcli dev wifi rescan", shell=True, check=True)
        time.sleep(2)  # Wait for scan to complete
        
        # Check if the network exists
        result = subprocess.run(
            f"nmcli -f SSID dev wifi list | grep '{ssid}'",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print(f"Error: Network '{ssid}' not found")
            return False
            
        # Update the preconfigured connection
        print("Updating WiFi configuration...")
        commands = [
            f'sudo nmcli connection modify "preconfigured" wifi.ssid "{ssid}"',
            f'sudo nmcli connection modify "preconfigured" wifi-sec.psk "{password}"',
            'sudo nmcli connection modify "preconfigured" connection.autoconnect yes',
            'sudo nmcli connection down "preconfigured"',
            'sudo nmcli connection up "preconfigured"'
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
        print("WiFi configuration updated successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 update_wifi.py <SSID> <PASSWORD>")
        print("Example: python3 update_wifi.py MyWiFiNetwork MyPassword123")
        sys.exit(1)
        
    ssid = sys.argv[1]
    password = sys.argv[2]
    
    if update_wifi_config(ssid, password):
        print("WiFi configuration completed successfully!")
    else:
        print("Failed to update WiFi configuration")
        sys.exit(1) 