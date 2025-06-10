#!/usr/bin/env python3
import subprocess
import sys
import time
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Command timed out")

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
        
        # First, try to bring down the connection with a timeout
        print("Disconnecting current connection...")
        down_result = run_command_with_timeout('sudo nmcli connection down "preconfigured"', timeout=5)
        if down_result is None:
            print("Warning: Could not disconnect current connection, proceeding anyway...")
        
        # Wait a moment before proceeding
        time.sleep(2)
        
        # Update the connection settings
        commands = [
            f'sudo nmcli connection modify "preconfigured" wifi.ssid "{ssid}"',
            f'sudo nmcli connection modify "preconfigured" wifi-sec.psk "{password}"',
            'sudo nmcli connection modify "preconfigured" connection.autoconnect yes'
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            result = run_command_with_timeout(cmd)
            if result is None:
                print(f"Warning: Command failed: {cmd}")
                continue
        
        # Try to bring up the connection
        print("Connecting to new network...")
        up_result = run_command_with_timeout('sudo nmcli connection up "preconfigured"', timeout=15)
        if up_result is None:
            print("Warning: Could not establish connection immediately")
            print("The connection will be attempted automatically")
        
        print("WiFi configuration updated successfully!")
        print("The system will attempt to connect to the new network")
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
        print("WiFi configuration completed successfully!")
        print("You may need to reboot for changes to take full effect")
    else:
        print("Failed to update WiFi configuration")
        sys.exit(1) 