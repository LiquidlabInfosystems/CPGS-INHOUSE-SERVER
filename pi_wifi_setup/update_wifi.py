#!/usr/bin/env python3
import subprocess
import sys
import time
import signal

def run_command_with_timeout(cmd, timeout=10):
    """Run a command with a timeout"""
    try:
        # Set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Run the command
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        # Disable the alarm
        signal.alarm(0)
        return result
    except TimeoutError:
        print(f"Command timed out after {timeout} seconds: {cmd}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return None
    finally:
        # Ensure alarm is disabled
        signal.alarm(0)

def timeout_handler(signum, frame):
    raise TimeoutError("Command timed out")

def update_wifi_config(ssid, password):
    """Update WiFi configuration on Raspberry Pi"""
    try:
        # First, check if the network exists
        print(f"\nScanning for WiFi network: {ssid}")
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
        print("\nUpdating WiFi configuration...")
        
        # Create a new connection profile for the new network
        print("Creating new connection profile...")
        create_cmd = f'sudo nmcli connection add type wifi con-name "preconfigured" ifname wlan0 ssid "{ssid}"'
        run_command_with_timeout(create_cmd)
        
        # Update the connection settings
        commands = [
            f'sudo nmcli connection modify "preconfigured" wifi-sec.key-mgmt wpa-psk',
            f'sudo nmcli connection modify "preconfigured" wifi-sec.psk "{password}"',
            'sudo nmcli connection modify "preconfigured" connection.autoconnect yes',
            'sudo nmcli connection modify "preconfigured" ipv4.method auto'
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            result = run_command_with_timeout(cmd)
            if result is None:
                print(f"Warning: Command failed: {cmd}")
                continue

        print("\nWiFi configuration updated successfully!")
        print("\n=== IMPORTANT: READ THESE INSTRUCTIONS ===")
        print("1. The new WiFi configuration has been saved")
        print("2. To apply the changes, you need to reboot:")
        print("   sudo reboot")
        print("\nAfter reboot:")
        print("1. The device will automatically connect to the new network")
        print("2. Connect to the new WiFi network on your computer")
        print("3. SSH into the device using the new IP address")
        print("\nCurrent IP address(es):")
        subprocess.run("ip addr show | grep 'inet ' | grep -v '127.0.0.1'", shell=True)
        print("\nRemember to update your SSH connection to use the new IP address after reconnecting!")
        
        return True
        
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
        print("\nWiFi configuration completed successfully!")
    else:
        print("\nFailed to update WiFi configuration")
        sys.exit(1)