import os
import cv2

class spaceViewStorage:
    def save(self, frame, slotIndex):
        """Save frame as a binary JPEG file."""
        cv2.imwrite(f'spaceViewOf:{slotIndex}.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])  # 85% quality

    def get(self,slotIndex):
        """Load the latest frame from disk."""
        return cv2.imread(f'spaceViewOf:{slotIndex}.jpg') if os.path.exists(f'spaceViewOf:{slotIndex}.jpg') else None


class licensePlateStorage:
    def save(self,frame, slotIndex):
        """Save frame as a binary JPEG file."""
        cv2.imwrite(f'licensePlateInSpace:{slotIndex}.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])  # 85% quality

    def get(self,slotIndex):
        """Load the latest frame from disk."""
        return cv2.imread(f'licensePlateInSpace:{slotIndex}.jpg') if os.path.exists(f'licensePlateInSpace:{slotIndex}.jpg') else None

