import cv2
import os

full_path = "assets/court_full.png"
img = cv2.imread(full_path)

H, W = img.shape[:2]
mid = W // 2

left = img[:, :mid]
right = img[:, mid:]

os.makedirs("assets", exist_ok=True)
cv2.imwrite("assets/court_left.png", left)
cv2.imwrite("assets/court_right.png", right)

print("Saved assets/court_left.png and assets/court_right.png")
