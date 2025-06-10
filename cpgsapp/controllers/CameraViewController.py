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
# Constants for car detection
CONFIDENCE_THRESHOLD = 0.5
CAR_CLASS_ID = 2  # COCO dataset class ID for car
IOU_THRESHOLD = 0.3  # Intersection over Union threshold for wrong parking detection
detection_running = False
last_detection_time = 0
DETECTION_INTERVAL = 1.0  # Time between detections in seconds


if IS_PI_CAMERA_SOURCE:
    from picamera2 import Picamera2
    Variables.cap = Picamera2()
    config = Variables.cap.create_preview_configuration(main={"size":(1280, 720)})
    Variables.cap.configure(config)
    Variables.cap.start()
    print("Using Pi Camera")
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



# Car detection model paths
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'yolov3-tiny.cfg')
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'yolov3-tiny.weights')
CLASSES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'coco.names')

# Initialize model variables
net = None
CLASSES = []
output_layers = []

def initialize_car_detection():
    """
    Initialize the car detection model and class labels.
    Returns True if initialization was successful, False otherwise.
    """
    global net, CLASSES, output_layers
    
    try:
        # Check if model files exist
        if not all(os.path.exists(path) for path in [CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH]):
            print("Error: One or more model files are missing. Please ensure all model files are present.")
            print(f"Config path: {CONFIG_PATH}")
            print(f"Weights path: {WEIGHTS_PATH}")
            print(f"Classes path: {CLASSES_PATH}")
            return False
            
        # Load class labels
        with open(CLASSES_PATH, 'r') as f:
            CLASSES = f.read().strip().split('\n')
        print(f"Loaded {len(CLASSES)} class labels")
            
        # Load the pre-trained YOLOv3-tiny model
        print("Loading YOLOv3-tiny model...")
        net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        
        # Set preferable backend and target
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        print(f"Model loaded successfully. Output layers: {output_layers}")
        return True
        
    except Exception as e:
        print(f"Error initializing car detection model: {str(e)}")
        return False

def is_wrong_parking(box, frame_shape):
    """
    Check if a car is parked incorrectly based on its position and orientation
    Returns True if the car is parked incorrectly, False otherwise
    """
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    
    # Calculate car dimensions and position
    car_width = x2 - x1
    car_height = y2 - y1
    car_center_x = (x1 + x2) / 2
    car_center_y = (y1 + y2) / 2
    
    # Calculate aspect ratio to determine car orientation
    aspect_ratio = car_width / car_height if car_height > 0 else 0
    
    print(f"\nAnalyzing car position:")
    print(f"Car dimensions: {car_width:.1f}x{car_height:.1f}")
    print(f"Car center: ({car_center_x:.1f}, {car_center_y:.1f})")
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    
    # Define parking rules
    wrong_parking = False
    reasons = []
    
    # Rule 1: Check if car is too close to the edge of the frame
    edge_margin = 50  # pixels
    if x1 < edge_margin or x2 > (width - edge_margin):
        wrong_parking = True
        reasons.append("Car too close to horizontal edge")
    
    # Rule 2: Check if car is parked at an angle (using aspect ratio)
    if aspect_ratio < 1.2 or aspect_ratio > 2.5:  # Normal cars have aspect ratio between 1.2 and 2.5
        wrong_parking = True
        reasons.append("Car appears to be parked at an angle")
    
    # Rule 3: Check if car is in the middle of the frame (should be in parking spot)
    frame_center_x = width / 2
    if abs(car_center_x - frame_center_x) > width * 0.3:  # Car should be within 30% of frame center
        wrong_parking = True
        reasons.append("Car not centered in parking spot")
    
    # Rule 4: Check if car is parked between two spaces
    # Calculate expected parking space width (assuming standard parking space width)
    expected_parking_width = width * 0.4  # Assuming parking space takes 40% of frame width
    if car_width > expected_parking_width * 1.3:  # If car is 30% wider than expected
        wrong_parking = True
        reasons.append("Car appears to be parked between two spaces")
        
        # Additional check for position relative to parking space boundaries
        parking_space_centers = [
            width * 0.25,  # Left parking space center
            width * 0.75   # Right parking space center
        ]
        
        # Find closest parking space center
        closest_center = min(parking_space_centers, key=lambda x: abs(x - car_center_x))
        if abs(car_center_x - closest_center) > expected_parking_width * 0.3:
            reasons.append("Car is not properly aligned with either parking space")
    
    if wrong_parking:
        print("\n=== WRONG PARKING DETECTED ===")
        print("Reasons:")
        for reason in reasons:
            print(f"- {reason}")
        print(f"Expected parking width: {expected_parking_width:.1f} pixels")
        print(f"Actual car width: {car_width:.1f} pixels")
    else:
        print("\nCar appears to be parked correctly")
    
    return wrong_parking



def start_car_detection():
    """
    Initialize and start the car detection system
    """
    global net, detection_running
    
    if not detection_running:
        print("\n=== Initializing Car Detection System ===")
        if initialize_car_detection():
            detection_running = True
            print("Car detection system initialized and running")
            return True
        else:
            print("Failed to initialize car detection system")
            return False
    return True

def detect_cars(frame):
    """
    Detect cars in the given frame using YOLOv3-tiny
    Returns list of car bounding boxes (x1, y1, x2, y2)
    """
    global net, output_layers, detection_running, last_detection_time
    
    current_time = time.time()
    
    # Check if we should run detection based on interval
    if current_time - last_detection_time < DETECTION_INTERVAL:
        return []  # Return empty list if not time for detection yet
    
    last_detection_time = current_time
    
    # If model is not initialized, try to initialize it
    if not detection_running:
        if not start_car_detection():
            return []
    
    try:
        print("\n=== Processing Frame ===")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
        # Convert grayscale to color if needed
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Resize frame to lower resolution for faster processing
        frame = cv2.resize(frame, (320, 200))
        
        height, width = frame.shape[:2]
        print(f"Frame size: {width}x{height}")
        
        # Get parking slot coordinates
        poslist = get_space_coordinates()
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input and forward pass
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        # Process detections
        car_boxes = []
        boxes = []
        confidences = []
        detection_count = 0
        
        # Process each output layer
        for output in outputs:
            if len(output.shape) != 2:
                continue
            
            for detection in output:
                try:
                    # Get scores for all classes
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = float(scores[class_id])
                    
                    if confidence > CONFIDENCE_THRESHOLD and class_id == CAR_CLASS_ID:
                        detection_count += 1
                        # Get box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate box corners
                        x1 = int(center_x - w/2)
                        y1 = int(center_y - h/2)
                        x2 = int(center_x + w/2)
                        y2 = int(center_y + h/2)
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        print(f"\nCar #{detection_count} detected:")
                        print(f"Position: ({x1}, {y1}) to ({x2}, {y2})")
                        print(f"Size: {w}x{h} pixels")
                        print(f"Confidence: {confidence:.2f}")
                        
                        # Check for wrong parking
                        if is_wrong_parking((x1, y1, x2, y2), frame.shape):
                            print("=== WRONG PARKING ALERT ===")
                            
                            # Find which slot the car is in
                            car_center = (center_x, center_y)
                            max_iou = 0
                            detected_slot = 0
                            
                            for slot_idx, slot_coords in enumerate(poslist):
                                # Convert slot coordinates to match resized frame
                                slot_x1 = int(slot_coords[0][0] * (320/1280))  # Scale from original 1280x720 to 320x200
                                slot_y1 = int(slot_coords[0][1] * (200/720))
                                slot_x2 = int(slot_coords[2][0] * (320/1280))
                                slot_y2 = int(slot_coords[2][1] * (200/720))
                                
                                # Calculate IOU between car and slot
                                slot_box = (slot_x1, slot_y1, slot_x2, slot_y2)
                                car_box = (x1, y1, x2, y2)
                                iou = calculate_iou(car_box, slot_box)
                                
                                if iou > max_iou:
                                    max_iou = iou
                                    detected_slot = slot_idx
                            
                            # Update MQTT server with wrong parking status for the detected slot
                            if max_iou > 0:  # Only update if we found a matching slot
                                print(f"Wrong parking detected in slot {detected_slot}")
                                update_server(detected_slot, "wrong parking", "")
                        
                        # Add to lists
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(confidence)
                except Exception as e:
                    print(f"Error processing detection: {str(e)}")
                    continue
        
        # Only apply NMS if we have detections
        if boxes and confidences:
            try:
                # Apply non-maximum suppression to remove overlapping boxes
                indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
                
                # Handle different types of indices
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten().tolist()
                elif isinstance(indices, tuple):
                    indices = list(indices)
                elif not isinstance(indices, list):
                    indices = [indices]
                
                # Add non-suppressed boxes to car_boxes
                for i in indices:
                    if isinstance(i, (list, tuple)):
                        i = i[0]  # Handle case where index is a list/tuple
                    car_boxes.append(tuple(boxes[i]))
                
                print(f"\nDetection Summary:")
                print(f"Total detections before NMS: {len(boxes)}")
                print(f"Final unique cars after NMS: {len(car_boxes)}")
                
            except Exception as e:
                print(f"Error in NMS: {str(e)}")
                # If NMS fails, return all boxes
                car_boxes = [tuple(box) for box in boxes]
                print("Using all boxes due to NMS error")
        else:
            print("\nNo cars detected in this frame")
        
        # Add a 5-second delay after detection
        time.sleep(1)
        
        return car_boxes
        
    except Exception as e:
        print(f"Error in car detection: {str(e)}")
        return []

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def detect_wrong_parking(car_boxes, parking_slots):
    """
    Detect wrong parking by checking if cars overlap with multiple parking slots
    Returns dictionary mapping slot indices to wrong parking status
    """
    wrong_parking_status = {}
    
    for slot_idx, slot in enumerate(parking_slots):
        slot_box = (slot[0][0], slot[0][1], slot[2][0], slot[2][1])
        overlapping_slots = set()
        
        for car_box in car_boxes:
            iou = calculate_iou(car_box, slot_box)
            if iou > IOU_THRESHOLD:
                overlapping_slots.add(slot_idx)
        
        if len(overlapping_slots) > 1:
            for slot_idx in overlapping_slots:
                wrong_parking_status[slot_idx] = "straddling"
    
    return wrong_parking_status

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
         return

    # Detect cars in the frame
    car_boxes = detect_cars(current_frame)
    
    # Detect wrong parking
    wrong_parking_status = detect_wrong_parking(car_boxes, poslist)

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
        # elif not object_detected and isLicensePlate:
        #      current_space_status = 'occupied'
        #      current_license_plate_payload = licensePlateBase64

        # Update confidence queue (using occupied status for pilot logic)
        Variables.CONFIDENCE_QUEUE[slotIndex].enqueue(current_space_status == 'occupied')

        # Get previous state for change detection
        prev_space_data = device_slot_data.get(str(device_id), {}).get(slotIndex)
        prev_space_status = prev_space_data['spaceStatus'] if prev_space_data else 'vacant'
        prev_license_plate = prev_space_data['licensePlate'] if prev_space_data else ''

        # Append current status for pilot light logic after the loop
        processed_slots_data.append({"slotIndex": slotIndex, "spaceStatus": current_space_status})

        # --- Only call update_server and save to DB if state has changed ---
        if current_space_status != prev_space_status :
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
            spaces = SpaceInfo.objects.all()
            Variables.pilotStatusofEachSpace = []
            for space in spaces:
                # Consider a space 'occupied' for pilot light if status is 'occupied' or 'obstacle detected'
                if space.space_status == "occupied" or space.space_status == "obstacle detected":
                    Variables.pilotStatusofEachSpace.append(True)
                else:
                    Variables.pilotStatusofEachSpace.append(False)

            # print(Variables.pilotStatusofEachSpace) # Uncomment if you want to see the list
            if(all(Variables.pilotStatusofEachSpace)):
                update_pilot('occupied')
            else:
                update_pilot('vacant') # Corrected typo
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

    # Detect cars in the frame
    car_boxes = detect_cars(current_frame)
    
    # Detect wrong parking
    wrong_parking_status = detect_wrong_parking(car_boxes, poslist)

    # Ensure confidence queues are initialized for all potential slots
    for slotIndex in range(Variables.TOTALSPACES):
        if len(Variables.CONFIDENCE_QUEUE) <= slotIndex:
             Variables.CONFIDENCE_QUEUE.append(FixedFIFO(CONSISTENCY_LEVEL))

    # Initialize or update in-memory space info based on DB or defaults
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
        slot_frame_gray_for_lp = slot_frame_gray.copy()
        slot_with_lp_drawing, licensePlate, isLicensePlate = dectect_license_plate(slot_frame_gray_for_lp)

        licensePlateBase64 = ""
        if isLicensePlate and licensePlate is not None and licensePlate.size > 0:
            licensePlateBase64 = image_to_base64(licensePlate)

        # Generate base64 for the slot frame with license plate drawing
        if isLicensePlate:
             spaceFrameBase64 = image_to_base64(slot_with_lp_drawing)
        else:
             spaceFrameBase64 = image_to_base64(slot_frame_color)

        # --- Obstacle Detection (Background Subtraction) ---
        background_img = backgrounds.get(str(device_id), {}).get(slotIndex)
        object_detected = False

        if background_img is not None:
            try:
                # Process background and current frame
                background_processed = preprocess_frame(background_img)
                slot_frame_processed = preprocess_frame(slot_frame_gray)

                # Ensure both frames have the same dimensions
                bg_height, bg_width = background_processed.shape
                slot_height, slot_width = slot_frame_processed.shape

                if bg_height != slot_height or bg_width != slot_width:
                    slot_frame_processed_resized = cv2.resize(slot_frame_processed, (bg_width, bg_height))
                else:
                    slot_frame_processed_resized = slot_frame_processed

                if slot_frame_processed_resized is not None:
                    diff = cv2.absdiff(background_processed, slot_frame_processed_resized)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    kernel = np.ones((5, 5), np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    object_detected = is_object_present(thresh)
            except Exception as e:
                print(f"Error in obstacle detection for slot {slotIndex}: {e}")

        # --- Determine Final Slot Status ---
        current_space_status = 'vacant'
        current_license_plate_payload = ''

        # Check for wrong parking first
        if slotIndex in wrong_parking_status:
            current_space_status = wrong_parking_status[slotIndex]
        elif object_detected and isLicensePlate:
            current_space_status = 'occupied'
            current_license_plate_payload = licensePlateBase64
        elif object_detected and not isLicensePlate:
            current_space_status = 'obstacle detected'
            current_license_plate_payload = ''

        # Update confidence queue
        Variables.CONFIDENCE_QUEUE[slotIndex].enqueue(current_space_status == 'occupied')

        # Get previous state for change detection
        prev_space_data = device_slot_data.get(str(device_id), {}).get(slotIndex)
        prev_space_status = prev_space_data['spaceStatus'] if prev_space_data else 'vacant'
        prev_license_plate = prev_space_data['licensePlate'] if prev_space_data else ''

        # Append current status for pilot light logic
        processed_slots_data.append({
            "slotIndex": slotIndex,
            "spaceStatus": current_space_status,
            "spaceFrame": spaceFrameBase64,
            "licensePlate": current_license_plate_payload
        })

        # --- Only call update_server and save to DB if state has changed ---
        if current_space_status != prev_space_status:
            print(f'Monitoring State change detected for slot {slotIndex}: Status {prev_space_status} -> {current_space_status}, LP change: {prev_license_plate != current_license_plate_payload}. Calling update_server...')
            update_server(slotIndex, current_space_status, current_license_plate_payload)

            # Update database if status changed
            try:
                space = SpaceInfo.objects.get(space_id=slotIndex)
                if space.space_status != current_space_status:
                    print(f'Monitoring Database status change detected for slot {slotIndex}: {space.space_status} -> {current_space_status}')
                    space.space_status = current_space_status
                    space.save()
            except SpaceInfo.DoesNotExist:
                print(f"Warning: SpaceInfo object not found for slotIndex {slotIndex} in monitoring. Cannot update database status.")
            except Exception as e:
                print(f"Error updating database for slot {slotIndex} in monitoring: {e}")

    # --- Pilot Update after processing all slots ---
    if IS_PI_CAMERA_SOURCE:
        try:
            spaces = SpaceInfo.objects.all()
            Variables.pilotStatusofEachSpace = []
            for space in spaces:
                # Consider a space 'occupied' for pilot light if status is 'occupied', 'obstacle detected', or 'straddling'
                if space.space_status in ["occupied", "obstacle detected", "straddling"]:
                    Variables.pilotStatusofEachSpace.append(True)
                else:
                    Variables.pilotStatusofEachSpace.append(False)

            if(all(Variables.pilotStatusofEachSpace)):
                update_pilot('occupied')
            else:
                update_pilot('vacant')
        except Exception as e:
            print(f"Error during pilot update after monitoring loop: {e}")

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

def stop_car_detection():
    """
    Stop the car detection system
    """
    global detection_running
    detection_running = False
    print("\n=== Car Detection System Stopped ===")

# End of Object detection functions :