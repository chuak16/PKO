import numpy as np
import win32gui, win32ui, win32con
from PIL import Image
from time import sleep
import time
import cv2 as cv
import os
import random
from ultralytics import YOLO
import pyautogui
import math

class WindowCapture:
    w = 0
    h = 0
    hwnd = None

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        self.window_x = window_rect[0] + self.cropped_x
        self.window_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        img = img[..., :3]
        img = np.ascontiguousarray(img)

        return img

    def get_window_position(self):
        return self.window_x, self.window_y

class ImageProcessor:
    model = None

    def __init__(self, weights_file):
        self.model = YOLO(weights_file)

    def process_image(self, img, target_classes=[0, 1], confidence_threshold=0.7):
        results = self.model(img)
        detections = results[0].boxes
        return [box for box in detections if box.conf[0] >= confidence_threshold and int(box.cls[0]) in target_classes]

    def draw_identified_objects(elf, img, detections, center_point):
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get box coordinates
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0]  # Get confidence
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)

            # Calculate target center
            target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = calculate_distance(center_point, target_center)

            # Draw bounding box and additional info
            cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv.putText(img, f'Class {class_id} {confidence:.2f}', (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv.putText(img, f'Distance: {distance:.2f}', (x1, y2 + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Show distance

        cv.imshow('Detection', img)
        cv.moveWindow('Detection',0,0)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

window_name = "Pirate King Online"
weights_file_name = r"best.pt or last.pt file director"
wincap = WindowCapture(window_name)
improc = ImageProcessor(weights_file_name)

while True:
    ss = wincap.get_screenshot()
    detections = improc.process_image(ss, target_classes=[0, 1])
    ss_with_boxes = ss.copy()

    # Calculate the center of the window relative to the screenshot
    center_x = wincap.w // 2  # Use the window dimensions for center
    center_y = wincap.h // 2
    center_point = (center_x, center_y)

    # Draw the reference center point (yellow dot)
    cv.circle(ss_with_boxes, center_point, 5, (0, 255, 255), -1)  # Yellow dot at reference center

    # Process the detections and draw them
    if detections:
        nearest_target = None
        min_distance = float('inf')

        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Get detection bounding box
            target_center = ((x1 + x2) / 2, (y1 + y2) / 2)  # Calculate target center
            distance = calculate_distance(center_point, target_center)  # Calculate distance from reference point

            # Draw the target center (red dot)
            cv.circle(ss_with_boxes, (int(target_center[0]), int(target_center[1])), 5, (0, 0, 255),
                      -1)  # Red dot at target center
            # Draw the distance text
            cv.putText(ss_with_boxes, f'Distance: {int(distance)}', (int(target_center[0]), int(target_center[1]) - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text for distance

            if distance < min_distance:
                min_distance = distance
                nearest_target = target_center

    # Show the window with detections at the top-left corner
    cv.imshow('Detection', ss_with_boxes)
    cv.moveWindow('Detection', 0, 0)  # Ensure it's placed at the top-left corner

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    sleep(0.2)

print('Finished.')
