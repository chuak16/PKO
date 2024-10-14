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


# WindowCapture class for capturing screenshots
class WindowCapture:
    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f'Window not found: {window_name}')

        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        border_pixels, titlebar_pixels = 8, 30
        self.w -= (border_pixels * 2)
        self.h -= (titlebar_pixels + border_pixels)
        self.cropped_x, self.cropped_y = border_pixels, titlebar_pixels
        self.window_x, self.window_y = window_rect[0] + self.cropped_x, window_rect[1] + self.cropped_y

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
        return np.ascontiguousarray(img)

    def get_window_position(self):
        return self.window_x, self.window_y


# ImageProcessor class for object detection and drawing boxes
class ImageProcessor:
    def __init__(self, weights_file):
        self.model = YOLO(weights_file)

    def process_image(self, img, target_classes=[0, 1], confidence_threshold=0.5):
        results = self.model(img)
        detections = results[0].boxes
        return [box for box in detections if box.conf[0] >= confidence_threshold and int(box.cls[0]) in target_classes]

    def draw_identified_objects(self, img, detections, center_point):
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get box coordinates
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0]  # Get confidence
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)

            # Calculate target center and distance
            target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = calculate_distance(center_point, target_center)

            # Draw bounding box and additional info
            cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv.putText(img, f'Class {class_id} {confidence:.2f}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv.putText(img, f'Distance: {distance:.2f}', (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                       2)

        # cv.imshow('Detection', img)
        # cv.moveWindow('Detection', 0, 0)


# HealthBarExtractor class for extracting and checking health bar status
class HealthBarExtractor:
    def __init__(self, window_capture):
        #230054 = 50% 234454 = 55% 225494 = 70% 243974 = 40% 251254 = 30%
        self.wincap = window_capture
        self.bar_x, self.bar_y, self.bar_width, self.bar_height = 48, 20, 135, 10
        self.fullhealth_value, self.depleted_value = 206140, 234454

    def extract_health_bar(self):
        img = self.wincap.get_screenshot()
        return img[self.bar_y:self.bar_y + self.bar_height, self.bar_x:self.bar_x + self.bar_width]

    def check_health_status(self):
        health_bar = self.extract_health_bar()
        health_bar = cv.cvtColor(health_bar, cv.COLOR_RGBA2BGR)
        redness_value = np.sum(health_bar[..., 2])

        if redness_value > self.depleted_value:
            return 'low'
        elif redness_value <= self.fullhealth_value:
            return 'full'
        else:
            return 'unknown'


# Utility function for distance calculation
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def emoji_making():
    number = random.randint(1, 100)
    if number == 1:
        pyautogui.hotkey('ctrl','1')
    elif number == 2:
        pyautogui.hotkey('ctrl','2')
    if number == 3:
        pyautogui.hotkey('ctrl','3')


# Main loop
def main():
    window_name = "Pirate King Online"
    weights_file_name = r"C:\Users\Kenny\PycharmProjects\yolo_bear\scripts\runs\detect\train3\weights\last.pt"
    wincap = WindowCapture(window_name)
    improc = ImageProcessor(weights_file_name)
    health_extractor = HealthBarExtractor(wincap)

    while True:
        ss = wincap.get_screenshot()
        detections = improc.process_image(ss, target_classes=[0, 1])
        emoji_making()
        # Center point reference for distance calculation
        center_point = (wincap.w // 2, wincap.h // 2)
        health_status = health_extractor.check_health_status()
        if health_status == 'low':
            sit_and_recover(health_extractor,wincap)

        if detections:
            click_on_nearest_target(wincap, detections, center_point)

        improc.draw_identified_objects(ss, detections, center_point)

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break

        sleep(0.2)


# Helper function for sitting down and recovering health
def sit_and_recover(health_extractor, wincap):
    # Sit down to start healing
    pyautogui.hotkey('insert')
    print("Low health - Sitting down")
    # Count how many times health is not full
    low_health_count = 0
    while True:
        time.sleep(5)  # Wait for 5 seconds (adjust as needed)

        # Check health condition
        health_status = health_extractor.check_health_status()
        if health_status == 'full':
            # Full health, stand up
            pyautogui.hotkey('insert')
            print("Full health - Standing up")
            sleep(1)
            break  # Exit the recovery loop

        # Health still low, increment count
        low_health_count += 1
        print(f"low health count: {low_health_count}")

        if low_health_count >= 6:
            # If health is low after 3 checks, stand up and run away
            pyautogui.hotkey('insert')  # Stand up
            print("Health still low after 3 checks - Running away")
            time.sleep(1)

            # Click on the middle top part of the window to run away
            window_x, window_y = wincap.get_window_position()
            top_middle_x = window_x + wincap.w // 2
            top_middle_y = window_y + 50  # Adjust the Y offset as needed for running away

            pyautogui.mouseDown(top_middle_x, top_middle_y)
            sleep(0.1)
            pyautogui.mouseUp()
            print(f'Clicked at ({top_middle_x}, {top_middle_y}) to run away')

            # Sit down again to heal after running away
            time.sleep(5)
            pyautogui.hotkey('insert')
            print("Sitting down again to heal")
            low_health_count = 0  # Reset the count after running away


# Helper function for clicking on the nearest target
def click_on_nearest_target(wincap, detections, center_point):
    nearest_target = None
    min_distance = float('inf')
    battle_time = 12
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        target_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = calculate_distance(center_point, target_center)

        if distance < min_distance:
            min_distance = distance
            nearest_target = target_center

    if nearest_target:
        click_x, click_y = nearest_target
        screen_click_x = int(click_x + wincap.get_window_position()[0])
        screen_click_y = int(click_y + wincap.get_window_position()[1])
        pyautogui.mouseDown(screen_click_x, screen_click_y)
        sleep(0.1)
        pyautogui.mouseUp()
        print(f'Clicked on target at: ({screen_click_x}, {screen_click_y})')
        sleep(battle_time)  # Delay before pressing 'ctrl + a'
        pyautogui.hotkey('ctrl', 'a')
        print("Pressed 'ctrl + a'")


if __name__ == '__main__':
    main()
