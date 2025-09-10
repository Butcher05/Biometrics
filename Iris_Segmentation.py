import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Image-path")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Remove bright ring light reflection
blurred = cv2.medianBlur(gray, 7)
_, bright_mask = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
gray_inpainted = cv2.inpaint(gray, bright_mask, 5, cv2.INPAINT_TELEA)

# Step 2: Detect pupil boundary (same as before)
pupil_mask = cv2.threshold(gray_inpainted, 60, 255, cv2.THRESH_BINARY_INV)[1]
pupil_edges = cv2.Canny(pupil_mask, 30, 60)
pupil_circles = cv2.HoughCircles(pupil_edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                 param1=100, param2=15, minRadius=20, maxRadius=90)

output = img.copy()

if pupil_circles is not None:
    pupil_circles = np.uint16(np.around(pupil_circles))
    x, y, r_pupil = pupil_circles[0, 0]
    cv2.circle(output, (x, y), r_pupil, (255, 0, 0), 2)  # Blue for pupil
else:
    raise Exception("Pupil not detected!")

# Step 3: Robust limbus detection using radial gradient
rows, cols = gray_inpainted.shape
Y, X = np.indices((rows, cols))
R = np.sqrt((X - x)**2 + (Y - y)**2)

# Create radial profile by averaging intensities over circles around the pupil
max_radius = min(rows, cols) // 2
radii = np.arange(r_pupil + 10, max_radius, 1)
profile = [np.mean(gray_inpainted[(R >= r - 1) & (R <= r + 1)]) for r in radii]
profile = np.array(profile)

# Find largest intensity drop = limbus edge
gradient = np.gradient(profile)
limbus_idx = np.argmax(np.abs(gradient))
r_limbus = radii[limbus_idx]

# Draw limbus boundary
cv2.circle(output, (x, y), int(r_limbus), (0, 255, 0), 2)  # Green for limbus

# Step 4: Show result
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Pupil (blue) and Limbus (green) Detection - Accurate")
plt.axis("off")
plt.show()
