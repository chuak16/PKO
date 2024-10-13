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

class HealthBarExtractor:
    def __init__(self, window_capture):
        bar_x = 48  # Adjust this
        bar_y = 20  # Adjust this
        bar_width = 135  # Adjust this
        bar_height = 10  # Adjust this
        self.wincap = window_capture
        self.bar_x = bar_x
        self.bar_y = bar_y
        self.bar_width = bar_width
        self.bar_height = bar_height

    def extract_health_bar(self):
        # Get screenshot from the game window
        img = self.wincap.get_screenshot()

        # Crop the health bar region (adjust bar_x, bar_y, bar_width, bar_height)
        health_bar = img[self.bar_y:self.bar_y + self.bar_height, self.bar_x:self.bar_x + self.bar_width]

        return health_bar

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def check_health_redness(health_bar):
    fullhealth_value = 206140
    depleted_value = 240054
    health_bar = cv.cvtColor(health_bar, cv.COLOR_RGBA2BGR)
    redness_value = np.sum(health_bar[..., 2])
    if redness_value > depleted_value:
        return 'low'
    elif redness_value <= fullhealth_value:
        return 'full'
    return 'unknown'

window_name = "Pirate King Online"
weights_file_name = r"C:\Users\Kenny\PycharmProjects\yolo_bear\scripts\runs\detect\train\weights\best.pt"
wincap = WindowCapture(window_name)
improc = ImageProcessor(weights_file_name)

while True:
    ss = wincap.get_screenshot()
    detections = improc.process_image(ss, target_classes=[0, 1])
    ss_with_boxes = ss.copy()
    center_x = wincap.w // 2  # Use the window dimensions for center
    center_y = wincap.h // 2
    center_point = (center_x, center_y)
    # Draw the reference center point (yellow dot)
    cv.circle(ss_with_boxes, center_point, 5, (0, 255, 255), -1)
    health_extractor = HealthBarExtractor(wincap)
    health_bar = health_extractor.extract_health_bar()
    health_status = check_health_redness(health_bar)
    if health_status == 'low':
        # Sit down to recover
        sleep(1)
        pyautogui.hotkey('insert')
        print("Low health - Sitting down")
        # Check health condition every 30 seconds until full health
        while health_status != 'full':
            time.sleep(5)  # Wait for 30 seconds
            ss = wincap.get_screenshot()
            health_bar = health_extractor.extract_health_bar()
            health_status = check_health_redness(health_bar)
        pyautogui.hotkey('insert')
        print("Full health - Standing up")
        sleep(1)

    if detections:
        # Find the nearest bear
        nearest_target = None
        min_distance = float('inf')

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            class_id = int(detection.cls[0])  # Get class ID
            confidence = detection.conf[0]  # Get confidence

            # Calculate target center for distance calculation
            target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance = calculate_distance(center_point, target_center)  # Calculate distance from reference point

            # Draw bounding box
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for class 0, red for others
            cv.rectangle(ss_with_boxes, (x1, y1), (x2, y2), color, 2)

            # Draw class label and confidence
            label = f"Class {class_id} {confidence:.2f}"
            cv.putText(ss_with_boxes, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw the distance text below the bounding box
            cv.putText(ss_with_boxes, f'Distance: {int(distance)}', (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)

            if distance < min_distance:
                    min_distance = distance
                    nearest_target = target_center

        if nearest_target:
            click_x, click_y = nearest_target
            # Convert target position relative to the game window
            window_pos = wincap.get_window_position()
            screen_click_x = int(click_x + window_pos[0])  # Adjust x based on window position
            screen_click_y = int(click_y + window_pos[1])  # Adjust y based on window position

            # Simulate mouse click on the nearest target
            pyautogui.mouseDown(screen_click_x, screen_click_y)
            time.sleep(0.05)  # Hold for 50 milliseconds
            pyautogui.mouseUp()
            print(f'Clicked on target at: ({screen_click_x}, {screen_click_y})')

            # Simulate 'ctrl + a' after a short delay
            time.sleep(15)
            pyautogui.hotkey('ctrl', 'a')
            print("Pressed 'ctrl + a'")
    # Show the detection window
    cv.imshow('Detection', ss_with_boxes)
    cv.moveWindow('Detection', 0, 0)

    health_bar = health_extractor.extract_health_bar()
    check_health_redness(health_bar)

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
    sleep(0.2)

print('Finished.')