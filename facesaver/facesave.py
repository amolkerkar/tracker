import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime


class FaceSaver:
    """
    This script takes an input image along with bounding box coordinates, 
    crops the specified region from the image, detects faces within that cropped region, 
    and saves the detected faces as separate image files in an output folder.
    """
    def __init__(self):
        pass

    def IDwisesave(self, frame, id, bounding_box, output_folder):
        def crop_image(image, bounding_box):
            x, y, w, h = bounding_box
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image

        def detect_faces(image):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_locations = face_recognition.face_locations(gray_image)
            return face_locations

        cropped_image = crop_image(frame, bounding_box)
        face_locations = detect_faces(cropped_image)
        
        id_folder = os.path.join(output_folder, str(id))
        if not os.path.exists(id_folder):
            os.makedirs(id_folder)
        
        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_image = cropped_image[top:bottom, left:right]
            cv2.imwrite(os.path.join(id_folder, f"face_{current_time}.jpg"), face_image)

        print("Faces saved successfully.")





