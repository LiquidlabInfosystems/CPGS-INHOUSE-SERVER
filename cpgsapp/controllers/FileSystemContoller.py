# Developed By Tecktrio At Liquidlab Infosystems
# Project: File System Contoller Methods
# Version: 1.0
# Date: 2025-03-08
# Description: A simple File System Controller to manage Files related activities

#Importing Functions
import json
from cpgsapp.models import SpaceInfo
import cv2
from storage import Variables


# Function to save the image 
def save_image(filename, image):
    # print('saving')
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes() 
    with open(f'storage/{filename}.jpg','wb') as file:
        file.write(image_bytes)
    return True


# helps in getting the latest space data
def get_space_info():
    try:
        with open("storage/spaceInfo.json", "r") as f:
            return json.loads(f.read().strip()) or {}
    except (json.JSONDecodeError, FileNotFoundError):
        print("Error reading JSON, using empty data.")
        return {}



# helps in updating the lastest space data
def update_space_info(new_data):
    try:
        with open("storage/spaceInfo.json", "w") as spaces:
            json.dump(new_data, spaces)
            return True
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        return False
    except FileNotFoundError:
        print("File not found or directory doesn't exist")
        return False



# Function to get to know which mode is being used (live/config)
def get_mode_info():
    with open('storage/mode.txt','r') as modeData:
        mode = modeData.read()
    return mode



# Function called to change from config to live
def change_mode_to_live():
    with open('storage/mode.txt','w') as modeData:
        modeData.write("live")



# Function called to change from live to config
def change_mode_to_config():
    with open('storage/mode.txt','w') as modeData:
        modeData.write("config")



# Function to save space coordinates
def save_space_coordinates(x, y):
    Variables.points.append((x, y))
    if len(Variables.points)%5 == 0:
        Variables.coordinates.append(Variables.points)
        with open('storage/coordinates.txt','w') as coordinate:
            SpaceInfo.objects.create(space_id = len(Variables.coordinates)-1, space_status = 'vacant')
            json.dump(Variables.coordinates, coordinate, indent=4)
        Variables.points = []



# Function to get space coordinates
def get_space_coordinates():
        with open('storage/coordinates.txt','r')as data:
            return json.load(data)
        


# To clear the space coordinates
def clear_space_coordinates():
    Variables.coordinates = []
    SpaceInfo.objects.all().delete()
    with open('storage/coordinates.txt','w') as coordinate:
        json.dump(Variables.coordinates, coordinate)
    Variables.points = []
    return True