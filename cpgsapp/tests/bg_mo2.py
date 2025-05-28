import cv2
import numpy as np

def preprocess_frame(frame):
    """Preprocess frame to reduce noise and lighting effects."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    gray = cv2.equalizeHist(gray)  # Normalize lighting
    return gray
cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
if not cap.isOpened():
    raise ValueError("Could not open camera. Check connection or index.")

# Capture and save background image
ret, frame = cap.read()
if ret:
    cv2.imwrite('storage/static_background.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    print("Background image saved.")
else:
    print("Failed to capture background frame.")
    exit()
# Load background image (manually captured)
background = cv2.imread('storage/static_background.jpg')
if background is None:
    raise ValueError("Background image not found!")
background_processed = preprocess_frame(background)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess current frame
    current_processed = preprocess_frame(frame)

    # Compute absolute difference (basic background subtraction)
    diff = cv2.absdiff(background_processed, current_processed)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Apply morphological ops to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Fill holes

    # Find contours and draw bounding boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    cv2.imshow("live feed", frame)
    cv2.imshow("background", background_processed)
    cv2.imshow("Foreground Mask", thresh)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()