# Developed By Tecktrio At Liquidlab Infosystems
# Project: Hardware Contoller Methods
# Version: 1.0
# Date: 2025-03-08
# Description: A simple Hardware Controller to Hardware systems like gpio related activities


# Importing functions
import subprocess
from . import FileSystemContoller
from cpgsserver.settings import IS_PI_CAMERA_SOURCE
from storage import Variables



# GPIO setup (only if running on a Raspberry Pi)
if IS_PI_CAMERA_SOURCE:
    from gpiozero import LED
    GREENLIGHT = LED(2)
    REDLIGHT = LED(3)
    MODEBUTTON = LED(4)
else:
    GREENLIGHT = REDLIGHT = None  # Avoid errors if running on non-RPi devices



# Function to set the pilot to green
def set_pilot_to_green():
    """Turn on the green light and turn off the red light."""
    if GREENLIGHT and REDLIGHT:
        GREENLIGHT.off()
        REDLIGHT.on()



# Function to set the pilot to red
def set_pilot_to_red():
    """Turn on the red light and turn off the green light."""
    # print('pilot changing')
    if GREENLIGHT and REDLIGHT:
        GREENLIGHT.on()
        REDLIGHT.off()

# Function to set the pilot to red
def set_pilot_to_off():
    """Turn off the red light and turn off the green light."""
    # print('pilot changing')
    if GREENLIGHT and REDLIGHT:
        GREENLIGHT.off()
        REDLIGHT.off()



# Function to update pilot light based on space availability
def update_pilot(mode):
    # """Update pilot light based on occupied spaces."""
    # spaces = FileSystemContoller.get_space_info()
    # if spaces != {}:
    #     if not spaces:
    #         print("No space data found. Defaulting to green light.")
    #         set_pilot_to_green()
    #         return
    #     occupied_count = sum(1 for space in spaces if space.get('spaceStatus') == 'occupied')
        # print("pilot updates",occupied_count)
        # available_spaces = Variables.TOTALSPACES - occupied_count
        if mode == 'occupied':
            set_pilot_to_red()
        elif mode == 'vaccant':
            set_pilot_to_green()
        else:
            set_pilot_to_off()


# helps in rebooting the system
def RebootSystem():
    subprocess.run("sudo reboot", shell=True, check=True)
