# Developed By Tecktrio At Liquidlab Infosystems
# Project: Network Contoller Methods
# Version: 1.0
# Date: 2025-03-08
# Description: A simple Network Controller to manage network related activities


# Importing functions
import json
import socket
import subprocess
from threading import Thread
import time
from cpgsapp.controllers.FileSystemContoller import get_space_info
from cpgsapp.models import Account, NetworkSettings, SpaceInfo
from cpgsapp.serializers import NetworkSettingsSerializer
from storage import Variables
import paho.mqtt.client as mqtt




class Broker:
    def __init__(self):
        self.NetworkSetting = NetworkSettings.objects.first()
        self.broker = self.NetworkSetting.server_ip
        self.port = self.NetworkSetting.server_port
        self.client = mqtt.Client()
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Start loop in the background to maintain network traffic
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("MQTT Broker Connected!")
            # Subscribe to commands topic with wildcard to receive all commands
            device_id = Account.objects.first().device_id
            command_topic = f"{device_id}/commands/#"
            client.subscribe(command_topic)
            print(f"Subscribed to command topic: {command_topic}")
            
            # Trigger publishing all statuses on connection (Wi-Fi reconnection)
            # print("Publishing all current statuses.")
            # publish_all_current_statuses() 
            
        else:
            print(f"Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        print("Disconnected from broker. Trying to reconnect...")
        # The loop_start handles reconnection attempts automatically, 
        # but you might want specific logic here if needed beyond basic retry.
        # self.reconnect() # Usually loop_start handles this, remove if not needed

    def _on_message(self, client, userdata, msg):
        """Callback for handling incoming MQTT messages."""
        try:
            # Decode and parse the message
            payload = msg.payload.decode()
            print(f"Message received on topic {msg.topic}: {payload}")
            
            # Try to parse as JSON
            try:
                command_data = json.loads(payload)
                if "action" in command_data:
                    command = command_data["action"]
                    
                    # Handle commands starting with $
                    if command.startswith('$'):
                        self._handle_dollar_command(command)
                    else:
                        print(f"Invalid command")
            except json.JSONDecodeError:
                print(f"Invalid JSON format in message: {payload}")
                
        except Exception as e:
            print(f"Error processing message: {e}")

    def _handle_dollar_command(self, command):
        """Handle commands that start with $"""
        try:
            # Remove the $ prefix
            cmd = command[1:].lower()
            
            # Handle different commands
            if cmd == 'gs':  # get status command
                print("Executing get status command")
                publish_all_current_statuses()
            else:
                print(f"Unknown command: ${cmd}")
                
        except Exception as e:
            print(f"Error handling dollar command: {e}")

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            print(f"Attempting to connect to broker {self.broker}:{self.port}")
        except Exception as e:
            print(f"Connection error: {e}")

    def reconnect(self):
        # This method is less critical if using loop_start, which handles auto-reconnect.
        # If loop_start is not enough, you might keep manual retry logic.
        def retry():
            while True:
                try:
                    self.client.reconnect()
                    print("Manual reconnect attempt successful.")
                    break
                except Exception as e:
                    print(f"Manual reconnect failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
        # Only start manual retry if loop_start's auto-reconnect is insufficient
        # Thread(target=retry, daemon=True).start()

    def send(self, topic, message):
        """Publish a message to a topic."""
        try:
            print(f"Attempting to publish to topic '{topic}'")
            result = self.client.publish(topic, message)
            # result.wait_for_publish() # Optional: block until publish completes
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("Message queued for publishing successfully.")
            else:
                print(f"Failed to queue message for publishing: {result.rc}")
            return result # Return result object if you need to check status later
        except Exception as e:
            print(f"Publish error: {e}")
            return None

    def disconnect(self):
        print("Disconnecting MQTT client...")
        self.client.loop_stop() # Stop the background loop
        self.client.disconnect()
        print("MQTT client disconnected.")

mainserverbroker = Broker()
mainserverbroker.connect()

# Helper funtin to chumk the data
def chunk_data(image_data, chunk_size):
    chunks = []
    for i in range(0, len(image_data), chunk_size):
        chunks.append(image_data[i:i+chunk_size])
    return chunks

# Dictionary structure: {device_id: {slotIndex: {slot_data}}}
device_slot_data = {}

def log_published_message(device_id, message_data):
    """Log the published message in a human-readable format."""
    print("\n" + "="*50)
    print(f"Publishing to device: {device_id}")
    print("-"*50)
    # Use json.dumps with indent for pretty printing
    print(json.dumps(message_data, indent=2))
    print("="*50 + "\n")


# THIS IS THE CUSTOM COMMAND FUNCTION (INVOKED BY "GET_STATUS")
def publish_all_current_statuses():
    """
    Retrieves the current status of all devices/slots from memory
    and publishes it to the server for each device.
    """
    print("Executing custom command: publish_all_current_statuses")
    
    if not device_slot_data:
        print("No device slot data available in memory.")

    for device_id, slots in device_slot_data.items():
        if not slots:
            print(f"No slots recorded for device {device_id}.")
            continue # Skip if a device has no slots initialized/updated

        message_data = {
            "deviceID": str(device_id),
            "slots": list(slots.values()) # Convert dictionary values to a list
        }

        # Log the message before sending
        log_published_message(device_id, message_data)

        # Send the message via MQTT
        topic = str(device_id) # Use device_id as the topic
        try:
            mainserverbroker.send(topic, json.dumps(message_data))
            print(f"Successfully published all statuses for device {device_id} to topic {topic}.")
        except Exception as e:
            print(f"Failed to publish all statuses for device {device_id}: {str(e)}")


# Function to initialize device_slot_data with default values
def initialize_device_slots_data():
    """Initializes device_slot_data with default 'vacant' statuses."""
    print("Initializing device slot data...")
    devices = Account.objects.all() # Get all devices from your database
    if not devices:
        print("No devices found in the database for initialization.")
        return

    for device in devices:
        device_id = device.device_id
        # Check if device already has some data (e.g., from a previous run state)
        if device_id not in device_slot_data:
             device_slot_data[device_id] = {}

        # Example: Initialize 3 slots per device if they don't exist
        for slot_index in range(0, 3): 
            if slot_index not in device_slot_data[device_id]:
                 device_slot_data[device_id][slot_index] = {
                    "slotIndex": slot_index,
                    "spaceStatus": "vacant", # Default status
                    "licensePlate": "" # Default empty
                }
    print("Device slot data initialization complete.")

def update_server(slotIndex, status, licenseplate):
    device_id = Account.objects.first().device_id # Assuming device ID is needed here
    
    # Ensure device entry exists
    if device_id not in device_slot_data:
         device_slot_data[device_id] = {}

    # Update the specific slot's data in the in-memory dictionary
    device_slot_data[device_id][slotIndex] = {
        "slotIndex": slotIndex,
        "spaceStatus": status,
        "licensePlate": licenseplate
    }
    
    print(f"Updated in-memory state for device {device_id}, slot {slotIndex}: status={status}, licensePlate='{licenseplate}'")

    # Now, publish the *entire* state for this device as requested
    publish_all_current_statuses() 


# helps in changing the hostname of the device
def change_hostname(new_hostname):
    try:
        current_hostname = subprocess.run(
            "hostname", shell=True, check=True, capture_output=True, text=True
        ).stdout.strip()
        if current_hostname == new_hostname:
            print(f"Hostname is already set to {new_hostname}. No changes required.")
            return True
        with open('/etc/hostname', 'w') as hostname_file:
            hostname_file.write(new_hostname)
        print(f"Updated /etc/hostname to {new_hostname}")
        with open('/etc/hosts', 'r') as hosts_file:
            hosts_content = hosts_file.readlines()
        with open('/etc/hosts', 'w') as hosts_file:
            for line in hosts_content:
                if line.startswith("127.0.1.1"):
                    hosts_file.write(f"127.0.1.1\t{new_hostname}\n")
                else:
                    hosts_file.write(line)
        print(f"Updated /etc/hosts with new hostname: {new_hostname}")
        subprocess.run(f"sudo hostnamectl set-hostname {new_hostname}", shell=True, check=True, capture_output=True, text=True)
        print(f"Hostname successfully changed to {new_hostname}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during hostname change process: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False



# Function to set a static IP
def set_static_ip(data):
    """Configures a static IP address."""
    try:
        nmcli_commands = [
            f"nmcli con modify {data['connection_name']} ipv4.addresses {data['static_ip']}",
            f"nmcli con modify {data['connection_name']} ipv4.gateway {data['gateway_ip']}",
            f"nmcli con modify {data['connection_name']} ipv4.dns {data['dns_ip']}",
            f"nmcli con modify {data['connection_name']} ipv4.method manual",
            f"nmcli con down {data['connection_name']}",
            f"nmcli con up {data['connection_name']}"
        ]
        for cmd in nmcli_commands:
            subprocess.run(cmd, shell=True, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting static IP: {e}")
        return False



# Function to set a dynamic IP
def set_dynamic_ip(data):
    """Configures a dynamic IP address."""
    try:
        nmcli_commands = [
            f"nmcli con modify {data['connection_name']} ipv4.method auto",
            f"nmcli con down {data['connection_name']}",
            f"nmcli con up {data['connection_name']}"
        ]
        for cmd in nmcli_commands:
            subprocess.run(cmd, shell=True, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting dynamic IP: {e}")
        return False



# Function to get network settings
def get_network_settings():
    """Retrieves the current network settings."""
    settings = NetworkSettings.objects.first()
    return NetworkSettingsSerializer(settings).data if settings else {}



# Function to save network settings
def saveNetworkSetting(new_settings):
    """Saves new network settings and applies them."""
    try:
        command = f"""
        nmcli con modify $(nmcli -g UUID con show --active | head -n 1) \
        ipv4.method manual 
        ipv4.addresses {new_settings.ipv4_address}/24 \
        ipv4.gateway {new_settings.gateway_address} \
        ipv4.dns "8.8.8.8 8.8.4.4"
        """
        subprocess.run(["sudo", "bash", "-c", command], check=True, text=True)
        connection_name = "preconfigured"
        subprocess.run(["sudo", "nmcli", "connection", "down", connection_name], check=True, text=True)
        subprocess.run(["sudo", "nmcli", "connection", "up", connection_name], check=True, text=True)
        subprocess.run(["sudo", "reboot", "now"], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error saving network settings: {e}")



# SCAN WIFI
def scan_wifi():
    """Scans for available WiFi networks and returns a list of SSIDs."""
    try:
        subprocess.run("sudo nmcli dev wifi rescan", shell=True, check=True, text=True)
        time.sleep(2)
        result = subprocess.run(
            "nmcli -f SSID dev wifi list",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        output_lines = result.stdout.strip().split('\n')[1:]
        ssids = list(set(line.strip() for line in output_lines if line.strip()))
        return ssids
    except subprocess.CalledProcessError as e:
        print(f"Error scanning WiFi networks: {e}")
        return []



# CONNEC TO THE WIFI
def connect_to_wifi(ssid, password):
    """Connects to a WiFi network and enables autoconnect after scanning."""
    print(f"Attempting to connect to WiFi: {ssid}")
    # Note: scan_wifi is called inside this function, adding delay.
    # Consider if you need to rescan every time or use cached results.
    available_ssids = scan_wifi() 
    if not available_ssids:
        print("No WiFi networks found or scanning failed.")
        return 401 # Return a relevant status code
        
    if ssid not in available_ssids:
        print(f"Error: Network '{ssid}' not found in scan results.")
        return 401 # Return a relevant status code

    try:
        # Note: This creates a new connection profile with the ssid name.
        # If you always want to use "preconfigured", you'd modify that instead.
        connect_cmd = f'sudo nmcli dev wifi connect "{ssid}" password "{password}"'
        # Using shell=True with user input can be dangerous. Sanitize inputs or avoid shell=True.
        result = subprocess.run(connect_cmd, shell=True, check=True, text=True, capture_output=True)
        print(f"Ready to Connect with WiFi: {ssid}")
        print(f"nmcli connect output:\n{result.stdout}")
        if result.stderr:
             print(f"nmcli connect stderr:\n{result.stderr}")
        
        # Assuming "preconfigured" is the connection profile you want to autoconnect on boot
        modify_cmd = f'sudo nmcli connection modify "preconfigured" connection.autoconnect yes'
        # Using shell=True here again, consider security.
        subprocess.run(modify_cmd, shell=True, check=True, text=True)
        print(f"Autoconnect enabled for preconfigured connection.")
        
        # Need to check actual connection status here, subprocess success might not mean connected.
        # You could check `nmcli device status` or try to ping the gateway/broker.
        print(f"Successfully initiated connection process to {ssid}.")
        return 200 # Indicate success attempt, actual connection might take time/fail later
     
    except subprocess.CalledProcessError as e:
        print(f"Error connecting to WiFi: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return 500 # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during connect_to_wifi: {e}")
        return 500 # Indicate failure


# --- Startup Logic ---
# 1. Initialize data structure (with default 'vacant' if needed)
initialize_device_slots_data()

# 2. Publish initial status (This is the boot-up trigger for the custom command)
print("Server starting up. Publishing initial device statuses.")
publish_all_current_statuses()

# 3. mainserverbroker instance is already created and connect() called above.
# The _on_connect callback will handle publishing on Wi-Fi reconnection and subscribing
# to the command topic.

# --- End Startup Logic ---


