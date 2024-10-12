import pygetwindow as gw
from PIL import ImageGrab
import time

# Get all windows with the same title
windows = gw.getWindowsWithTitle('Pirate King Online')
"""
for window in windows:
    print(f"Window Title: {window.title}, hwnd: {window._hWnd}, "
          f"Location: ({window.left}, {window.top}), "
          f"Size: ({window.width}x{window.height})")
"""
# Find the specific window by handle
window = next(w for w in windows if w._hWnd == 263732)

# Capture screenshot every second, save to folder
for i in range(151, 200):
    bbox = window.left, window.top, window.right, window.bottom
    screenshot = ImageGrab.grab(bbox)
    screenshot.save(f'C:\\Users\\Kenny\\PycharmProjects\\yolo_bear\\data\\train\\images_2\\screen_{i}.png')
    time.sleep(1)
    print("screenshot")
print("done")
