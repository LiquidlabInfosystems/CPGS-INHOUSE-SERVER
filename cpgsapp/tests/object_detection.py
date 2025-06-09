import cv2
import numpy as np

def preprocess_frame(frame):
    """Preprocess frame to reduce noise and lighting effects."""
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

def is_object_present(thresh, contour_area_threshold=300, pixel_change_threshold=10):
    """Determine if an object is present based on pixel change and contour analysis."""
    # Rule 1: Significant pixel change
    change_percent = (np.sum(thresh == 255) / thresh.size) * 100
    enough_pixel_change = change_percent > pixel_change_threshold
    
    # Rule 2: Physical contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > contour_area_threshold]
    has_contours = len(large_contours) > 0
    
    return enough_pixel_change and has_contours

cap = cv2.VideoCapture(0)  # Try 0 if 1 doesn't work
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
background = cv2.imread('storage/static_background.jpg')
if background is None:
    raise ValueError("Background image not found!")
background_processed = preprocess_frame(background)

# Create a black image for stats display
stats_display = np.zeros((400, 400, 3), dtype=np.uint8)  # Fixed 400x400 square

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess current frame
    current_processed = preprocess_frame(frame)

    # Compute absolute difference and get stats
    diff_stats = calculate_frame_difference_stats(background_processed, current_processed)
    
    # Compute thresholded difference for display
    diff = cv2.absdiff(background_processed, current_processed)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Apply morphological ops to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Check if object is present using the combined method
    object_detected = is_object_present(thresh)
    
    # Find contours and draw bounding boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update stats display
    stats_display.fill(0)
    y_offset = 30
    line_height = 25
    
    # Add object detection status to stats
    detection_status = "OBJECT DETECTED" if object_detected else "No object"
    cv2.putText(stats_display, f"Status: {detection_status}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    stats_text = [
        f"Mean Difference: {diff_stats['mean_diff']:.2f}",
        f"Std Deviation: {diff_stats['std_diff']:.2f}",
        f"Max Difference: {diff_stats['max_diff']}",
        f"Changed Pixels: {diff_stats['changed_pixels']}",
        f"Change Percentage: {diff_stats['change_percentage']:.2f}%"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(stats_display, text, (10, y_offset + (i+1)*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    # Display detection status on live feed
    cv2.putText(frame, detection_status, (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if object_detected else (0, 255, 0), 2)

    # Display results
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Background", background_processed)
    cv2.imshow("Foreground Mask", thresh)
    cv2.imshow("Frame Difference Stats", stats_display)
    cv2.imshow("Raw Difference", diff)  # Show the raw difference image

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()