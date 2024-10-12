import numpy as np
import win32gui, win32ui, win32con
from PIL import Image
from time import sleep
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

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while (True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpeg")
            sleep(1)

    def get_window_size(self):
        return (self.w, self.h)


class ImageProcessor:
    model = None

    def __init__(self, weights_file):
        self.model = YOLO(weights_file)

    def process_image(self, img, confidence_threshold=0.55, target_class = 0):
        results = self.model(img)
        detections = results[0].boxes
        return [box for box in detections if box.conf[0] >= confidence_threshold and int(box.cls[0]) == target_class]

    def draw_identified_objects(self, img, detections):
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get box coordinates
            class_id = int(box.cls[0])  # Get class ID
            confidence = box.conf[0]  # Get confidence

            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv.putText(img, f'Class {class_id} {confidence:.2f}', (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.imshow('Detection', img)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
# Run this cell to initiate detections using the trained model.

window_name = "Pirate King Online"
weights_file_name = r"C:\Users\Kenny\PycharmProjects\yolo_bear\scripts\runs\detect\train17\weights\last.pt"

wincap = WindowCapture(window_name)
improc = ImageProcessor(weights_file_name)

while True:
    ss = wincap.get_screenshot()
    detections = improc.process_image(ss)
    # Draw bounding boxes on the image
    # ss_with_boxes = ss.copy()  # Make a copy of the image to draw on
    # improc.draw_identified_objects(ss_with_boxes, detections)
    # cv.imshow("Window", ss_with_boxes)
    if detections:
        # Calculate the center of the window
        window_pos = wincap.get_window_position()
        center_x = window_pos[0] + wincap.w // 2
        center_y = window_pos[1] + wincap.h // 2
        center_point = (center_x, center_y)

        # Find the nearest bear
        nearest_bear = None
        min_distance = float('inf')

        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            bear_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance = calculate_distance(center_point, bear_center)

            if distance < min_distance:
                min_distance = distance
                nearest_bear = bear_center

        if nearest_bear:
            # Click on the nearest bear
            click_x = int(nearest_bear[0]) + window_pos[0]
            click_y = int(nearest_bear[1]) + window_pos[1]
            pyautogui.mouseDown(click_x, click_y)
            sleep(0.05)  # Hold for 50 milliseconds
            pyautogui.mouseUp()
            print(f'Clicked on nearest bear at: ({click_x}, {click_y})')
            sleep(random.uniform(3, 5))
            sleep(2)
            pyautogui.hotkey('ctrl', 'a')
            print("pressed ctrl + a")# Optional: Wait for half a second after pressing Ctrl + A

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
    sleep(0.2)

print('Finished.')