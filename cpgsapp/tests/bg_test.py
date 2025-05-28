import cv2
import numpy as np
import time

# Initialize camera
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

# Load background image
BACKGROUND_IMAGE = cv2.imread('storage/static_background.jpg', cv2.IMREAD_GRAYSCALE)
if BACKGROUND_IMAGE is None:
    raise ValueError("Background image not loaded. Check file path.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Debug: Show raw camera feed
    # cv2.imshow("Raw Camera Feed", frame)
    cv2.imshow("Background model", BACKGROUND_IMAGE)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Frame", gray_frame)

    # Resize background if needed
    if BACKGROUND_IMAGE.shape != gray_frame.shape:
        BACKGROUND_IMAGE = cv2.resize(BACKGROUND_IMAGE, (gray_frame.shape[1], gray_frame.shape[0]))

    # Compute difference and threshold
    diff = cv2.absdiff(BACKGROUND_IMAGE, gray_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    # cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", thresh_cleaned)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()