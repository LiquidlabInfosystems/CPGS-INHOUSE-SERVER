import cv2
import numpy as np

def preprocess_frame(frame):
    """Preprocess frame to reduce noise and lighting effects."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    gray = cv2.equalizeHist(gray)  # Normalize lighting
    return gray

# Initialize camera
cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
if not cap.isOpened():
    raise ValueError("Could not open camera. Check connection or index.")

# Create background subtractor objects (try either MOG2 or KNN)
# backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
backSub = cv2.createBackgroundSubtractorKNN(history=10000, dist2Threshold=400, detectShadows=False)

# Learning rate (0 means don't update background model, 1 means fully replace)
learning_rate = 0.001

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess current frame (optional - background subtractors work well without this)
    # current_processed = preprocess_frame(frame)
    current_processed = frame
    
    # Apply background subtraction
    fg_mask = backSub.apply(current_processed, learningRate=learning_rate)
    
    # Remove shadow values (value=127 in MOG2/KNN)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Apply morphological ops to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    
    # Find contours and draw bounding boxes
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    cv2.imshow("Live Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Pause background learning when space is pressed
        learning_rate = 0
    elif key == ord('r'):  # Resume background learning when 'r' is pressed
        learning_rate = 0.001

cap.release()
cv2.destroyAllWindows()