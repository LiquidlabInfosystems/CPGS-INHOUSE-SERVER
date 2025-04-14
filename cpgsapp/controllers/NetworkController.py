# Developed By Tecktrio At Liquidlab Infosystems
# Project: Network Contoller Methods
# Version: 1.0
# Date: 2025-03-08
# Description: A simple Network Controller to manage network related activities


# Importing functions
import json
import socket
import subprocess
import time
from cpgsapp.controllers.FileSystemContoller import get_space_info
from cpgsapp.models import NetworkSettings, SpaceInfo
from cpgsapp.serializers import NetworkSettingsSerializer
from storage import Variables
import paho.mqtt.client as mqtt


NetworkSetting = NetworkSettings.objects.first()
broker = NetworkSetting.server_ip  # You can use your own broker
port = NetworkSetting.server_port
client = mqtt.Client()
client.connect(broker, port, keepalive=60)

# Helper funtin to chumk the data
def chunk_data(image_data, chunk_size):
    chunks = []
    for i in range(0, len(image_data), chunk_size):
        chunks.append(image_data[i:i+chunk_size])
    return chunks



# Update the main server when there is a dectectin in the monitoring spaces
# def update_server(spaceId, status, licenseplate):
#     """Detects changes in space status and updates the main server."""
#     NetworkSetting = NetworkSettings.objects.first()
#     data_to_send = {
#         "spaceID": spaceId,
#         "spaceStatus": status,
#         "licensePlate": licenseplate
#     }
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     CHUNK_SIZE = 20
#     MESSAGE = f"{data_to_send}".encode()    
#     chunks = chunk_data(MESSAGE, CHUNK_SIZE)
#     for chuck in chunks:
#         sock.sendto(chuck, (NetworkSetting.server_ip, NetworkSetting.server_port))
#     sock.close()
#     print("Dectection update send to server - spaceID : ",data_to_send['spaceID'], ", status : ",data_to_send['spaceStatus'], ", server Address : ", NetworkSetting.server_ip, ":", NetworkSetting.server_port)


def update_server(spaceId, status, licenseplate):
    
    # NetworkSetting = NetworkSettings.objects.first()
    data_to_send = {
        "spaceID": spaceId,
        "spaceStatus": status,
        "licensePlate": licenseplate
    }
    topic = "cpgs"
    message = json.dumps(data_to_send)

    try:
        client.publish(topic, message)
        print("Message published successfully.")
        # client.disconnect()
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
        ipv4.method manual \
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


