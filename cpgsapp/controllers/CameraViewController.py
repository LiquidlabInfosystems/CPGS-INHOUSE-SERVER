# Developed By Tecktrio At Liquidlab Infosystems
# Project: Camera Contoller Methods
# Version: 1.0
# Date: 2025-03-08
# Description: A simple Camera Controller to manage camera related activities

# Importing functions
import base64
import json
import time
from cpgsapp.models import SpaceInfo, Account, DeviceBackground
from cpgsapp.utils import FixedFIFO
import cv2
import numpy as np
from cpgsapp.controllers.FileSystemContoller import get_space_coordinates, get_space_info, update_space_info
from cpgsapp.controllers.HardwareController import  update_pilot
from cpgsapp.controllers.NetworkController import update_server, device_slot_data
from cpgsserver.settings import CONFIDENCE_LEVEL, CONSISTENCY_LEVEL, IS_PI_CAMERA_SOURCE
from storage import Variables
import os
import glob
from cpgsapp.controllers.Device_id_config import get_device_id
# from paddleocr import PaddleOCR

# Initialize PaddleOCR once at module level
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
# Camera Input Setup


if IS_PI_CAMERA_SOURCE:
    from picamera2 import Picamera2
    Variables.cap = Picamera2()
    config = Variables.cap.create_preview_configuration(main={"size":(1280, 720)})
    Variables.cap.configure(config)
    Variables.cap.start()
else: 
    Variables.cap = cv2.VideoCapture(1)



def image_to_base64(frame):
    try:
        frame_contiguous = np.ascontiguousarray(frame)
        success, encoded_img = cv2.imencode('.jpg', frame_contiguous)
        if not success:
            print("Failed to encode frame to JPEG")
            return None
        image_bytes = encoded_img.tobytes()
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_string}"
        return data_url
    except Exception as e:
        print(f"Error converting frame to base64: {str(e)}")
        return None
    


# helps in decting license plate in the current frame
def dectect_license_plate(space):
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml') 
    isLicensePlate = False
    license_plate = None
    plates = plate_cascade.detectMultiScale(space, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
    for (x, y, w, h) in plates:
        isLicensePlate = True 
        cv2.rectangle(space, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        license_plate = space[y:y+h, x:x+w]
    return space, license_plate, isLicensePlate



# Function called for calibrating 
async def video_stream_for_calibrate():
    while True:
        frame  = load_camera_view()
        with open('coordinates.txt','r')as data:
            for space_coordinates in json.load(data):
                    for index in range (0,len(space_coordinates)-1):
                        x1 = int(space_coordinates[index][0])
                        y1 = int(space_coordinates[index][1])
                        x2 = int(space_coordinates[index+1][0])
                        y2 = int(space_coordinates[index+1][1])    
                        cv2.line(frame,(x1,y1),(x2,y2), (0, 255, 0), 2)  
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')
        readyToSendFrame = f"data:image/jpeg;base64,{encoded_frame}"
        yield readyToSendFrame



# helps in capturing the frame from physical camera
def capture():
    """Synchronous capture function optimized for performance."""
    if IS_PI_CAMERA_SOURCE:
        frame = Variables.cap.capture_array()
        if frame is None:
            print("Failed to capture frame from PiCamera")
            time.sleep(0.5)
    else:
        ret, frame = Variables.cap.read()
        if not ret:
            print("Failed to capture frame from VideoCapture")
            time.sleep(0.1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if frame.size > 0:
        frame = cv2.resize(frame, (1280 , 720))
        return frame
    else:
        print("Invalid frame received")
    # time.sleep(.8)



# LOAD CAMERA VIEW 
def load_camera_view(max_attempts=5, delay=0.05):
        camera_view = capture()
        if camera_view is not None and not camera_view.size == 0: 
            return camera_view
        else:
            height, width = 720 , 1280 
            blank_image = np.zeros((height, width, 1), dtype=np.uint8)
            return blank_image
      


# Function called for getting the camera view with space coordinates
def get_camera_view_with_space_coordinates():
    frame = load_camera_view()
    with open('storage/coordinates.txt', 'r') as data:
        for space_coordinates in json.load(data):
            for index in range (0,len(space_coordinates)-1):
                x1 = int(space_coordinates[index][0])
                y1 = int(space_coordinates[index][1])
                x2 = int(space_coordinates[index+1][0])
                y2 = int(space_coordinates[index+1][1])    
                cv2.line(frame,(x1,y1),(x2,y2), (255, 255, 255), 3)  
        points = Variables.points
        if len(points)>1:
            for index in range (0,len(points)-1):
                x1 = int(points[index][0])
                y1 = int(points[index][1])
                x2 = int(points[index+1][0])
                y2 = int(points[index+1][1])    
                cv2.line(frame,(x1,y1),(x2,y2), (255, 255, 255), 3)  
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    return frame_bytes



#Function called to detect license plate
def getSpaceMonitorWithLicensePlateDectection(x, y, w, h ):
        camera_view = load_camera_view()
        space_view = camera_view[y:y+h, x:x+w]
        licensePlateinSpace,licensePlate, isLicensePlate =  dectect_license_plate(space_view)
        licensePlateinSpaceInBase64 = image_to_base64(licensePlateinSpace)
        # for space in Variables.SPACES:
        # if space['slotIndex'] == slotIndex:
        licensePlateBase64 = ""
        if isLicensePlate:
            # licensePlateStorage.save(frame=licensePlate,slotIndex=slotIndex)
            licensePlateBase64 = image_to_base64(licensePlate)
            # licenseNumber = get_ocr(licensePlateBase64)
            # print(licenseNumber)
            # licensePlateStorage.update_base64(image_to_base64(Variables.licensePlate))
        #     space['spaceStatus'] = "occupied"
        # space['spaceFrame'] = Variables.licensePlateinSpaceInBase64
        # spaceFrameStorage.update_base64(image_to_base64(Variables.licensePlateinSpace))
        # spaceViewStorage.save(frame=licensePlateinSpace,slotIndex=slotIndex)
        # space['licensePlate'] = Variables.licensePlateBase64
        # update_space_info(Variables.SPACES)
        return isLicensePlate,licensePlateBase64, licensePlateinSpaceInBase64



# Function to start live mode and detect the available license plates
def liveMode():
    '''
    SCAN the parking slot FOR VEHICLE and use object detection for each slot.
    '''
    # Load backgrounds once per call
    backgrounds = load_background_images_for_all_devices()
    poslist = get_space_coordinates()
    Variables.TOTALSPACES = len(poslist)

    # Capture the current frame ONCE before processing all slots
    current_frame = load_camera_view()

    # Check if frame capture failed or returned an empty frame
    if current_frame is None or current_frame.size == 0:
         print("Failed to capture a valid frame for live mode monitoring.")
         return # Return without processing if no frame captured

    # Ensure confidence queues are initialized for all potential slots
    for slotIndex in range(Variables.TOTALSPACES):
        if len(Variables.CONFIDENCE_QUEUE) <= slotIndex:
             Variables.CONFIDENCE_QUEUE.append(FixedFIFO(CONSISTENCY_LEVEL))

    # Access device_slot_data from NetworkController for status comparison and update


    device_id = get_device_id()

    # List to collect processed slot data for pilot light logic
    processed_slots_data = []

    # Ensure device_id exists in device_slot_data before accessing
    if device_id not in device_slot_data:
         device_slot_data[device_id] = {}

    for slotIndex, pos in enumerate(poslist):
        # --- Frame Cropping and Initial Processing ---
        SpaceCoordinates = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]], [pos[2][0], pos[2][1]], [pos[3][0], pos[3][1]]])
        pts = np.array(SpaceCoordinates, np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        # Ensure bounding box coordinates are valid and within frame dimensions
        frame_height, frame_width = current_frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        if w <= 0 or h <= 0:
             print(f"Skipping slot {slotIndex} due to invalid or zero-sized bounding box: x={x}, y={y}, w={w}, h={h}")
             processed_slots_data.append({"slotIndex": slotIndex, "spaceStatus": 'vacant'})
             continue

        slot_frame_crop = current_frame[y:y+h, x:x+w]

        if len(slot_frame_crop.shape) == 3:
             slot_frame_gray = cv2.cvtColor(slot_frame_crop, cv2.COLOR_BGR2GRAY)
        else:
             slot_frame_gray = slot_frame_crop

        # --- License Plate Detection ---
        slot_frame_gray_for_lp = slot_frame_gray.copy()
        slot_with_lp_drawing, licensePlate, isLicensePlate = dectect_license_plate(slot_frame_gray_for_lp)

        licensePlateBase64 = ""
        if isLicensePlate and licensePlate is not None and licensePlate.size > 0:
            licensePlateBase64 = image_to_base64(licensePlate)

        # --- Obstacle Detection (Background Subtraction) ---
        background_img = backgrounds.get(str(device_id), {}).get(slotIndex)
        object_detected = False

        if background_img is not None:
            background_processed = preprocess_frame(background_img)
            slot_frame_processed = preprocess_frame(slot_frame_gray)

            # --- Ensure sizes match before calculating difference ---
            bg_height, bg_width = background_processed.shape[:2]
            sf_height, sf_width = slot_frame_processed.shape[:2]

            slot_frame_processed_resized = None
            if bg_height != sf_height or bg_width != sf_width:
                 try:
                     slot_frame_processed_resized = cv2.resize(slot_frame_processed, (bg_width, bg_height))
                 except cv2.error as e:
                     print(f"Error resizing slot frame for slot {slotIndex} in liveMode: {e}")
            else:
                 slot_frame_processed_resized = slot_frame_processed

            if slot_frame_processed_resized is not None:
                diff = cv2.absdiff(background_processed, slot_frame_processed_resized)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                object_detected = is_object_present(thresh)

        # --- End Obstacle Detection ---

        # --- Determine Final Slot Status ---
        current_space_status = 'vacant'
        current_license_plate_payload = ''

        if object_detected and isLicensePlate:
             current_space_status = 'occupied'
             current_license_plate_payload = licensePlateBase64
        elif object_detected and not isLicensePlate:
             current_space_status = 'obstacle detected'
             current_license_plate_payload = ''
        elif not object_detected and isLicensePlate:
             current_space_status = 'occupied'
             current_license_plate_payload = licensePlateBase64

        # Update confidence queue (using occupied status for pilot logic)
        Variables.CONFIDENCE_QUEUE[slotIndex].enqueue(current_space_status == 'occupied')

        # Get previous state for change detection
        prev_space_data = device_slot_data.get(str(device_id), {}).get(slotIndex)
        prev_space_status = prev_space_data['spaceStatus'] if prev_space_data else 'vacant'
        prev_license_plate = prev_space_data['licensePlate'] if prev_space_data else ''

        # Append current status for pilot light logic after the loop
        processed_slots_data.append({"slotIndex": slotIndex, "spaceStatus": current_space_status})

        # --- Only call update_server and save to DB if state has changed ---
        if current_space_status != prev_space_status or current_license_plate_payload != prev_license_plate:
             print(f'Live Mode State change detected for slot {slotIndex}: Status {prev_space_status} -> {current_space_status}, LP change: {prev_license_plate != current_license_plate_payload}. Calling update_server...')
             update_server(slotIndex, current_space_status, current_license_plate_payload)

             # Update database if status changed
             try:
                 space = SpaceInfo.objects.get(space_id=slotIndex)
                 if space.space_status != current_space_status:
                     print(f'Live Mode Database status change detected for slot {slotIndex}: {space.space_status} -> {current_space_status}')
                     space.space_status = current_space_status
                     space.save()
             except SpaceInfo.DoesNotExist:
                  print(f"Warning: SpaceInfo object not found for slotIndex {slotIndex} in liveMode. Cannot update database status.")
             except Exception as e:
                 print(f"Error updating database for slot {slotIndex} in liveMode: {e}")

    # --- Pilot Update after processing all slots ---
    if IS_PI_CAMERA_SOURCE:
        try:
            any_occupied = any(s['spaceStatus'] == 'occupied' for s in processed_slots_data)
            if any_occupied:
                 update_pilot('occupied')
            else:
                 update_pilot('vacant')
        except Exception as e:
            print(f"Error during pilot update after liveMode loop: {e}")

    # liveMode runs continuously and does not return a value to a request handler.



# Function used to monitor the spaces
def get_monitoring_spaces():
    '''
    SCAN the parking slot FOR VEHICLE and use object detection for each slot.
    '''
    # Load backgrounds once per call
    backgrounds = load_background_images_for_all_devices()
    poslist = get_space_coordinates()
    Variables.SPACES = [] # This seems unused after object detection integration. Consider removing if confirmed.
    Variables.TOTALSPACES = len(poslist)

    # Capture the current frame ONCE before processing all slots
    current_frame = load_camera_view()

    # Check if frame capture failed or returned an empty frame
    if current_frame is None or current_frame.size == 0:
         print("Failed to capture a valid frame for monitoring.")
         return [] # Return empty list if no frame captured

    # Ensure confidence queues are initialized for all potential slots
    for slotIndex in range(Variables.TOTALSPACES):
        if len(Variables.CONFIDENCE_QUEUE) <= slotIndex:
             Variables.CONFIDENCE_QUEUE.append(FixedFIFO(CONSISTENCY_LEVEL))

    # Initialize or update in-memory space info based on DB or defaults
    # This call with an empty list might not be doing what's intended for initializing device_slot_data
    # The NetworkController.initialize_device_slots_data() function seems more appropriate for initial setup.
    # Consider reviewing the purpose of this specific update_space_info call here.
    # update_space_info(Variables.SPACES) # Needs review if it's intended to initialize/update DB space info.

    # Dynamically get device_id for this system
    # Access device_slot_data from NetworkController for status comparison
    # from cpgsapp.controllers.NetworkController import device_slot_data

    device_id = get_device_id()  # Use the first device_id by default

    # List to collect processed slot data for the return value
    processed_slots_data = []

    # Ensure device_id exists in device_slot_data before accessing (for status comparison)
    if device_id not in device_slot_data:
         device_slot_data[device_id] = {}

    for slotIndex, pos in enumerate(poslist):
        # --- Frame Cropping and Initial Processing ---
        SpaceCoordinates = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]], [pos[2][0], pos[2][1]], [pos[3][0], pos[3][1]]])
        pts = np.array(SpaceCoordinates, np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        # Ensure bounding box coordinates are valid and within frame dimensions
        frame_height, frame_width = current_frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        # Ensure width and height don't go past frame boundaries
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        if w <= 0 or h <= 0:
             print(f"Skipping slot {slotIndex} due to invalid or zero-sized bounding box: x={x}, y={y}, w={w}, h={h}")
             # Append default vacant data for this slot so it's included in the response structure
             processed_slots_data.append({
                 "slotIndex": slotIndex,
                 "spaceStatus": 'vacant', # Assume vacant if cannot process
                 "spaceFrame": '', # Add empty spaceFrame for invalid slots
                 "licensePlate": ''
             })
             continue # Skip processing this slot if bounding box is invalid

        # Crop the current frame for this slot
        slot_frame_crop = current_frame[y:y+h, x:x+w]

        # Ensure slot_frame_crop is grayscale for subsequent processing like license plate detection
        # and for background subtraction comparison
        if len(slot_frame_crop.shape) == 3:
             slot_frame_gray = cv2.cvtColor(slot_frame_crop, cv2.COLOR_BGR2GRAY)
             # Also keep a color version if you need to draw on it later for spaceFrame
             slot_frame_color = slot_frame_crop.copy()
        else:
             slot_frame_gray = slot_frame_crop # Already grayscale or single channel
             # Create a color version for drawing if needed
             slot_frame_color = cv2.cvtColor(slot_frame_crop, cv2.COLOR_GRAY2BGR)


        # --- License Plate Detection ---
        # The function dectect_license_plate expects a grayscale image.
        # It modifies the image in place to draw the rectangle.
        # Pass a copy if you need the original grayscale crop later without drawings.
        # However, for spaceFrame, we need the drawing, so we'll use a copy of the grayscale
        # and then potentially draw on the color version. Let's simplify and draw on a copy
        # of the color frame *after* determining status, if we need a visual indicator.
        # For now, just detect LP on a copy of the gray frame.
        slot_frame_gray_for_lp = slot_frame_gray.copy()
        slot_with_lp_drawing, licensePlate, isLicensePlate = dectect_license_plate(slot_frame_gray_for_lp)

        licensePlateBase64 = ""
        if isLicensePlate and licensePlate is not None and licensePlate.size > 0:
            # Generate base64 for the cropped license plate itself if detected and valid
            licensePlateBase64 = image_to_base64(licensePlate)

        # Generate base64 for the slot frame with license plate drawing
        # We need to draw on a color image for better visualization if the original was grayscale.
        # Let's draw the LP bounding box on the slot_frame_color copy.
        if isLicensePlate:
             # Need to re-calculate LP bounding box relative to the original slot_frame_color if drawing here
             # Or modify dectect_license_plate to also return the drawing on a color image.
             # For simplicity now, let's use the slot_with_lp_drawing result (which is grayscale)
             # and convert it to base64. If you need color drawings, dectect_license_plate needs to be updated.
             spaceFrameBase64 = image_to_base64(slot_with_lp_drawing) # Using the grayscale image with drawing
        else:
             # If no license plate, just use the base64 of the original slot frame crop (color or gray converted to color)
             spaceFrameBase64 = image_to_base64(slot_frame_color) # Using the color version (or gray converted to color)


        # --- Obstacle Detection (Background Subtraction) ---
        background_img = backgrounds.get(str(device_id), {}).get(slotIndex)
        object_detected = False
        # obstacleFrameBase64 = '' # Variable to store base64 of frame with obstacle drawing if needed

        if background_img is not None:
            # Both background_img and slot_frame_gray are grayscale here.
            # Preprocess them (gaussian blur, equalize hist)
            background_processed = preprocess_frame(background_img) 
            slot_frame_processed = preprocess_frame(slot_frame_gray)

            # --- Ensure sizes match before calculating difference ---
            # Get dimensions of the background image (after preprocessing)
            bg_height, bg_width = background_processed.shape[:2]
            sf_height, sf_width = slot_frame_processed.shape[:2]

            # Only resize if dimensions don't match
            if bg_height != sf_height or bg_width != sf_width:
                 try:
                     # Resize slot_frame_processed to match the background size
                     slot_frame_processed_resized = cv2.resize(slot_frame_processed, (bg_width, bg_height))
                     # print(f"Resized slot frame for slot {slotIndex} from {sf_width}x{sf_height} to match background {bg_width}x{bg_height}.") # Uncomment for debugging
                 except cv2.error as e:
                     print(f"Error resizing slot frame for slot {slotIndex}: {e}")
                     # If resize fails, cannot perform diff, treat as no object detected for this slot
                     object_detected = False
                     slot_frame_processed_resized = None # Set to None to avoid using invalid data
            else:
                 slot_frame_processed_resized = slot_frame_processed # No resizing needed

            # Only proceed with diff calculation if resizing was successful or not needed
            if slot_frame_processed_resized is not None:
                # Calculate difference using size-matched processed frames
                diff = cv2.absdiff(background_processed, slot_frame_processed_resized)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                # Apply morphological operations to clean up the thresholded image
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Opening to remove small objects/noise
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Closing to fill small holes

                # Determine if an object is present based on the thresholded image
                # This function checks pixel change and contour area
                object_detected = is_object_present(thresh)

                # Optional: Draw obstacle contours on a color version of the original slot crop for visualization
                # if object_detected:
                #     # Need to find contours on the potentially resized threshold image (thresh)
                #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #     # To draw on the original slot_frame_color, contour coordinates would need scaling
                #     # For simplicity, let's generate base64 of the thresh image itself or draw on the resized processed frame.
                #     # If you need drawings on the original color frame, the logic here needs to be more complex (scaling contours).
                #     # As an alternative, we can draw on the original slot_frame_color based on the bounding box of the *entire* slot where the object was detected.
                #     # But the requirement is to show the "spaceFrame" which is the slot crop.
                #     # Let's stick to providing the slot frame with LP drawing (spaceFrameBase64 calculated earlier) for spaceFrame key.
                #     pass # Drawing logic moved/simplified


        # --- End Obstacle Detection ---

        # --- Determine Final Slot Status based on Obstacle and License Plate Detection ---
        current_space_status = 'vacant' # Default status
        current_license_plate_payload = '' # Default empty license plate payload

        if object_detected and isLicensePlate:
             # If both obstacle and license plate are detected, it's an occupied slot
             current_space_status = 'occupied'
             current_license_plate_payload = licensePlateBase64 # Include detected LP
        elif object_detected and not isLicensePlate:
             # If obstacle detected but no license plate, it's an obstacle
             current_space_status = 'obstacle detected'
             current_license_plate_payload = '' # No license plate for obstacles
        elif not object_detected and isLicensePlate:
             # If no obstacle detected but license plate found, treat as occupied
             current_space_status = 'occupied'
             current_license_plate_payload = licensePlateBase64 # Include detected LP
        # else (not object_detected and not isLicensePlate): status remains 'vacant'

        # Update confidence queue based on whether it's considered occupied for pilot light logic
        # Enqueue True if status is 'occupied', False otherwise.
        Variables.CONFIDENCE_QUEUE[slotIndex].enqueue(current_space_status == 'occupied')

        # Recalculate confidence levels after enqueueing
        queue = Variables.CONFIDENCE_QUEUE[slotIndex].get_queue()
        Occupied_count = queue.count(True)
        Vaccency_count = queue.count(False)
        # Confidence levels are not directly used to set the main status in this revised logic,
        # but are kept for potential future use or pilot light logic if it uses confidence.
        Occupied_confidence = int((Occupied_count/CONSISTENCY_LEVEL)*100)
        Vaccency_confidence = int((Vaccency_count/CONSISTENCY_LEVEL)*100)

        # --- End Determine Final Slot Status ---

        # Get the previous status from in-memory data for change detection
        prev_space_data = device_slot_data.get(device_id, {}).get(slotIndex)
        # Use 'vacant' as default if slot data doesn't exist yet (first run)
        prev_space_status = prev_space_data['spaceStatus'] if prev_space_data else 'vacant'
        # Also get previous license plate to see if it changed
        prev_license_plate = prev_space_data['licensePlate'] if prev_space_data else ''


        # Create the slot data dictionary for the current state (for return value)
        slot_data = {
            "slotIndex": slotIndex,
            "spaceStatus": current_space_status,
            "spaceFrame": spaceFrameBase64, # Include the base64 image of the slot frame (with LP drawing if detected)
            "licensePlate": current_license_plate_payload # Use the determined payload license plate
        }

        # Append the processed slot data to the list for the return value
        processed_slots_data.append(slot_data)

        # --- Only call update_server and save to DB if status or license plate has changed ---
        # This prevents continuous sending when status and LP are stable
        # Check if either status or the *payload* license plate has changed
        if current_space_status != prev_space_status or current_license_plate_payload != prev_license_plate:
             print(f'Status or License Plate change detected for slot {slotIndex}: Status {prev_space_status} -> {current_space_status}, LP change: {prev_license_plate != current_license_plate_payload}. Calling update_server...')

             # Update the in-memory state and send via MQTT using the existing function
             # The update_server function handles updating device_slot_data and publishing.
             update_server(slotIndex, current_space_status, current_license_plate_payload) # Pass the LP payload

             # Update database if status changed (only status stored in SpaceInfo model)
             # Note: License plate is not stored in SpaceInfo based on your model. Only update status if it changed.
             # Fetch the SpaceInfo object for the current slot if needed for DB update
             try:
                 space = SpaceInfo.objects.get(space_id=slotIndex)
                 if space.space_status != current_space_status:
                     print(f'Database status change detected for slot {slotIndex}: {space.space_status} -> {current_space_status}')
                     space.space_status = current_space_status
                     space.save()
             except SpaceInfo.DoesNotExist:
                  print(f"Warning: SpaceInfo object not found for slotIndex {slotIndex}. Cannot update database status.")
             except Exception as e:
                 print(f"Error updating database for slot {slotIndex}: {e}")
        # else: No action needed if both status and license plate are stable


    # --- Pilot Update after processing all slots ---
    # If pilot status depends on the status of ALL slots, update it once after the loop.
    # This is a more efficient location for the pilot update logic if needed.
    if IS_PI_CAMERA_SOURCE:
        try:
            # Pilot logic: Pilot is occupied if ANY slot is 'occupied'. It's vacant only if ALL are 'vacant'.
            # Using the 'processed_slots_data' list for the most recent computed statuses.
            any_occupied = any(s['spaceStatus'] == 'occupied' for s in processed_slots_data)

            if any_occupied:
                 update_pilot('occupied') # Pilot is 'occupied' if any space is occupied
            else:
                 update_pilot('vacant') # Pilot is 'vacant' only if all spaces are vacant
        except Exception as e:
            print(f"Error during pilot update after loop: {e}")
    # --- End Pilot Update ---

    # Return the list of processed slot data
    # The MQTT sending is handled by update_server within the loop, only on change.
    return processed_slots_data


def capture_and_save_background_mask(device_id):
    """
    Captures background images and saves them to database 
    """
    frame = capture()
    if frame is None:
        print("Failed to capture frame.")
        return False

    coordinates_list = get_space_coordinates()
    for slot_index, slot_coords in enumerate(coordinates_list):
        pts = np.array(slot_coords, np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        cropped_frame = frame[y:y+h, x:x+w]
        
        # Convert to JPEG bytes
        _, buffer = cv2.imencode('.jpg', cropped_frame)
        jpg_bytes = buffer.tobytes()
        
        # Save to database
        DeviceBackground.objects.update_or_create(
            device_id=device_id,
            slot_index=slot_index,
            defaults={'background_image': jpg_bytes}
        )
        print(f"Saved background for device {device_id} slot {slot_index}")
    return True

def capture_and_save_background_mask_for_all_device_ids():
    """
    Fetches all device IDs from the Account model and calls capture_and_save_all_backgrounds for each.
    """
    device_ids = list(Account.objects.values_list('device_id', flat=True))
    for device_id in device_ids:
        print(f"Capturing backgrounds for device: {device_id}")
        capture_and_save_background_mask(device_id)
    return True

def load_background_images_for_all_devices():
    """
    Loads all background images for each device and slot index from the database.
    Returns a dictionary: { device_id: { slot_index: image (numpy array) } }
    """
    device_backgrounds = {}
    
    # Query all device backgrounds from the database
    backgrounds = DeviceBackground.objects.all().select_related()
    
    for bg in backgrounds:
        try:
            # Convert BinaryField to numpy array
            jpg_bytes = bytes(bg.background_image)
            nparr = np.frombuffer(jpg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Failed to decode image for device {bg.device_id} slot {bg.slot_index}")
                continue
                
            # Add to dictionary
            if bg.device_id not in device_backgrounds:
                device_backgrounds[bg.device_id] = {}
            device_backgrounds[bg.device_id][bg.slot_index] = img
            
        except Exception as e:
            print(f"Error processing background for device {bg.device_id} slot {bg.slot_index}: {e}")
            continue
    
    return device_backgrounds


# Object detection functions :

def preprocess_frame(frame):
    """Preprocess frame to reduce noise and lighting effects."""
    # If frame is already grayscale, don't convert
    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    gray = cv2.equalizeHist(gray)  # Normalize lighting
    return gray

def calculate_frame_difference_stats(background, current):
    """
    Calculate statistical differences between background and current frame.
    Returns dictionary containing various difference metrics.
    """
    diff = cv2.absdiff(background, current)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    stats = {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'max_diff': np.max(diff),
        'changed_pixels': np.sum(thresh == 255),
        'change_percentage': (np.sum(thresh == 255) / thresh.size * 100)
    }
    return stats

def is_object_present(thresh, contour_area_threshold=500, pixel_change_threshold=30):
    """Determine if an object is present based on pixel change and contour analysis."""
    # Rule 1: Significant pixel change
    change_percent = (np.sum(thresh == 255) / thresh.size) * 100
    enough_pixel_change = change_percent > pixel_change_threshold
    
    # Rule 2: Physical contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > contour_area_threshold]
    has_contours = len(large_contours) > 0
    
    return enough_pixel_change and has_contours


# End of Object detection functions :