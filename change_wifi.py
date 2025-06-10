#!/usr/bin/env python3
import sys
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cpgsserver.settings')
django.setup()

from cpgsapp.controllers.NetworkController import connect_to_wifi, scan_wifi
from cpgsapp.models import NetworkSettings

def change_wifi_configuration(ssid, password):
    """Change WiFi configuration using existing NetworkController functionality"""
    try:
        # First scan for available networks
        print("Scanning for available WiFi networks...")
        available_networks = scan_wifi()
        
        if not available_networks:
            print("No WiFi networks found!")
            return False
            
        print("\nAvailable networks:")
        for network in available_networks:
            print(f"- {network}")
            
        if ssid not in available_networks:
            print(f"\nError: Network '{ssid}' not found in available networks")
            return False
            
        print(f"\nAttempting to connect to {ssid}...")
        status = connect_to_wifi(ssid, password)
        
        if status == 200:
            # Update NetworkSettings in database
            network_settings = NetworkSettings.objects.first()
            if network_settings:
                network_settings.ap_ssid = ssid
                network_settings.ap_password = password
                network_settings.save()
                print("\nWiFi configuration updated successfully!")
                print("The new configuration will be used after reboot")
                return True
        elif status == 401:
            print("\nError: Network not found or connection failed")
        else:
            print("\nError: Failed to connect to WiFi network")
            
        return False
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 change_wifi.py <SSID> <PASSWORD>")
        print("Example: python3 change_wifi.py MyWiFiNetwork MyPassword123")
        sys.exit(1)
        
    ssid = sys.argv[1]
    password = sys.argv[2]
    
    if change_wifi_configuration(ssid, password):
        print("\nWiFi configuration completed successfully!")
    else:
        print("\nFailed to update WiFi configuration")
        sys.exit(1) 