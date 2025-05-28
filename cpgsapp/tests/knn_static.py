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

# Capture and store static background
print("Capturing background image...")
ret, background = cap.read()
if not ret:
    raise ValueError("Failed to capture background frame")
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
cv2.imwrite('storage/static_background.jpg', background_gray)
print("Background image saved.")

# Initialize KNN with the static background
backSub = cv2.createBackgroundSubtractorKNN(history=1, dist2Threshold=400, detectShadows=False)

# Manually set the background model
backSub.apply(background_gray, learningRate=0)

# Learning rate (0 means don't update background model)
learning_rate = 0  # Keep background static by default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply background subtraction with static background
    fg_mask = backSub.apply(current_gray, learningRate=learning_rate)
    
    # Threshold mask
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display
    cv2.imshow("Live Detection", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Static Background", background_gray)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):  # Toggle background updating
        learning_rate = 0.001 if learning_rate == 0 else 0
    elif key == ord('c'):  # Capture new background
        ret, background = cap.read()
        if ret:
            background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            backSub.apply(background_gray, learningRate=1.0)
            cv2.imwrite('background.jpg', background_gray)
            print("Background updated!")

cap.release()
cv2.destroyAllWindows()