import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
if not cap.isOpened():
    raise ValueError("Could not open camera. Check connection or index.")

# Create MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=False)

# Variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction to grayscale frame
    fg_mask = bg_subtractor.apply(gray_frame, learningRate=-1)
    
    # Post-processing to remove noise and small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Closing to fill small holes
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Opening to remove small noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours on the processed mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Calculate and display FPS
    frame_count += 1
    if frame_count >= 30:  # Update FPS every 30 frames
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        frame_count = 0
        start_time = end_time
    
    # Display grayscale frame in the window

    
    # Add FPS text to the original frame
    cv2.putText(gray_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Grayscale Input", gray_frame)
    # Display results
    # cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Background Model", bg_subtractor.getBackgroundImage())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()