import cv2 as cv
import numpy as np
import win32gui, win32ui, win32con
import pyautogui
import os
from PIL import Image
from time import sleep

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

def check_health_redness(health_bar):
    fullhealth_value = 206140
    depleted_value = 251254
    health_bar = cv.cvtColor(health_bar, cv.COLOR_RGBA2BGR)
    redness_value = np.sum(health_bar[..., 2])
    if redness_value > depleted_value:
        pyautogui.hotkey('ins')
        print("sit down")
    elif redness_value <= fullhealth_value:
        sleep(2)
        pyautogui.hotkey('insert')
        print("fullhealth")

    print(f'Redness value: {redness_value}')

# WindowCapture class as defined by you
wincap = WindowCapture("Pirate King Online")

# Initialize HealthBarExtractor
health_extractor = HealthBarExtractor(wincap)

# Extract and check the health bar status
health_bar = health_extractor.extract_health_bar()
check_health_redness(health_bar)

# Display the extracted health bar for verification
cv.imshow("Health Bar", health_bar)
cv.waitKey(0)
cv.destroyAllWindows()
