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
        
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        
        # Start loop in the background to maintain network traffic
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # Call this at server startup
            initialize_and_publish_all_slots()
            print("Broker Connected!")
        else:
            print(f"Connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from broker. Trying to reconnect...")
        self.reconnect()

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
        except Exception as e:
            print(f"Connection error: {e}")

    def reconnect(self):
        def retry():
            while True:
                try:
                    self.client.reconnect()
                    break
                except:
                    print("Reconnect failed. Retrying in 5 seconds...")
                    time.sleep(5)
        Thread(target=retry, daemon=True).start()

    def send(self, topic, message):
        try:
            result = self.client.publish(topic, message)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print("Message published successfully.")
            else:
                print(f"Failed to publish message: {result.rc}")
        except Exception as e:
            print(f"Publish error: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

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

def initialize_and_publish_all_slots():
    """Initialize all devices/slots with default values and publish."""
    devices = Account.objects.all()
    for device in devices:
        device_id = device.device_id
        device_slot_data[device_id] = {}
        
        # Initialize slots (example: 10 slots per device)
        for slot_index in range(0, 3):
            device_slot_data[device_id][slot_index] = {
                "slotIndex": slot_index,
                "spaceStatus": "vacant",
                "licensePlate": ""
            }
        
        # Prepare message
        message = {
            "deviceID": str(device_id),
            "slots": list(device_slot_data[device_id].values())
        }
        
        # Print the message being published
        print(f"Publishing to device {device_id}:")
        print(json.dumps(message, indent=2))
        
        try:
            mainserverbroker.send(str(device_id), json.dumps(message))
            print(f"Successfully published to {device_id}")
        except Exception as e:
            print(f"Failed to publish to {device_id}: {str(e)}")



def update_server(slotIndex, status, licenseplate):
    device_id = Account.objects.first().device_id
    topic = str(device_id)

    # Initialize device entry if not exists
    if device_id not in device_slot_data:
        device_slot_data[device_id] = {}

    # Update or create the slot data
    device_slot_data[device_id][slotIndex] = {
        "slotIndex": slotIndex,
        "spaceStatus": status,
        "licensePlate": licenseplate
    }

    # Prepare message with ALL slots for this device
    message = {
        "deviceID": str(device_id),
        "slots": list(device_slot_data[device_id].values())
    }

    print(f"Publishing slots for device {device_id}:\n{json.dumps(message, indent=2)}")

    try:
        mainserverbroker.send(topic, json.dumps(message))
        print(f"All slots published successfully for device {device_id}")
    except socket.error as e:
        print(f"Failed to connect to MQTT broker: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



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
    available_ssids = scan_wifi()
    if not available_ssids:
        print("No WiFi networks found or scanning failed.")
        return 401
    if ssid not in available_ssids:
        print(f"Error: Network '{ssid}' not found in scan results.")
        return 401
    try:
        connect_cmd = f'sudo nmcli dev wifi connect "{ssid}" password "{password}"'
        result = subprocess.run(connect_cmd, shell=True, check=True, text=True, capture_output=True)
        print(f"Ready to Connect with WiFi: {ssid}")
        modify_cmd = f'sudo nmcli connection modify "preconfigured" connection.autoconnect yes'
        subprocess.run(modify_cmd, shell=True, check=True, text=True)
        print(f"Autoconnect enabled for {ssid}")
     
    except subprocess.CalledProcessError as e:
        print(f"Error connecting to WiFi: {e}")


