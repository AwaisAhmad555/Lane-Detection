import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Helper Functions
# ----------------------------------------

def region_of_interest(image):
    """
    Apply polygon mask to keep only region where lanes are expected.
    """
    height, width = image.shape
    mask = np.zeros_like(image)

    polygon = np.array([[ 
        (int(0.03 * width), height),         # Bottom-left
        (int(0.97 * width), height),         # Bottom-right
        (int(0.5 * width), int(0.35 * height)) # Top-center
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, mask

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))

    lane_lines = []
    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        lane_lines.append(make_coordinates(image, left_avg))
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        lane_lines.append(make_coordinates(image, right_avg))

    return lane_lines

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 200, 200), 8)  # Dark yellow
    return line_image

def fill_lane_region(image, lines, lane_status):
    lane_overlay = np.zeros_like(image)
    height, width = image.shape[:2]

    if len(lines) == 2:
        left_line, right_line = lines[0], lines[1]
        color = (0, 140, 255) if lane_status == "Current Lane" else (0, 0, 255)

        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], np.int32)
        cv2.fillPoly(lane_overlay, [pts], color)

        # Optional: Opposite lane (extend beyond right line)
        width = image.shape[1]
        opposite_pts = np.array([
            [right_line[0], right_line[1]],
            [right_line[2], right_line[3]],
            [width, right_line[3]],
            [width, right_line[1]]
        ], np.int32)

        # cv2.fillPoly(lane_overlay, [opposite_pts], (0, 140, 255))  # Orange = opposite lane


    return lane_overlay


# ----------------------------------------
# Image Lane Detection
# ----------------------------------------

image = cv2.imread('lane 21.png')  # Replace with your image path
# image = cv2.resize(image, (960, 540))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
cropped_edges, mask = region_of_interest(edges)

lineGap = 15 #5                    adjust for better result

lines = cv2.HoughLinesP(
    cropped_edges,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=lineGap
)

averaged_lines = average_slope_intercept(image, lines)

# Determine lane status
cx = image.shape[1] // 10
lane_status = "Unknown"
try:
    left_line = averaged_lines[0]
    if left_line[0] > cx:
        lane_status = "Current Lane"
    else:
        lane_status = "Opposite Lane"
except:
    lane_status = "Unknown"

line_image = display_lines(image, averaged_lines)
lane_area = fill_lane_region(image, averaged_lines, lane_status)

combined = cv2.addWeighted(image, 0.7, lane_area, 0.3, 0)
final_output = cv2.addWeighted(combined, 0.9, line_image, 1, 1)

# Overlay text
overlay = final_output.copy()
cv2.rectangle(overlay, (20, 20), (400, 80), (50, 50, 50), -1)
alpha = 0.6
final_output = cv2.addWeighted(overlay, alpha, final_output, 1 - alpha, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 180, 255) if lane_status == "Current Lane" else (0, 0, 255)
cv2.putText(final_output, f"Lane Status: {lane_status}", (40, 60), font, 0.8, text_color, 2, cv2.LINE_AA)

# ----------------------------------------
# Display with Matplotlib (2x3 Grid)
# ----------------------------------------

plt.figure(figsize=(18, 10))

# 1 - Original Image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 2 - Grayscale + Blur (display grayscale)
plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

# 3 - Canny Edges
plt.subplot(2, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis('off')

# 4 - ROI Mask
plt.subplot(2, 3, 4)
plt.imshow(mask, cmap='gray')
plt.title("ROI Mask")
plt.axis('off')

# 5 - Masked Edges
plt.subplot(2, 3, 5)
plt.imshow(cropped_edges, cmap='gray')
plt.title("Masked Edges")
plt.axis('off')

# 6 - Final Output
plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
plt.title("Lane Detection Output")
plt.axis('off')

# ----------------------------------------
# Save Matplotlib 2x3 Grid as Image
# ----------------------------------------
plt.tight_layout()
plt.show()
grid_output_path = "lane_detection_grid.png"
# plt.savefig(grid_output_path, dpi=300, bbox_inches='tight')
# plt.close()
print(f"Lane detection grid saved at: {grid_output_path}")


# ----------------------------------------
# Save Results Instead of Displaying
# ----------------------------------------

output_path = "lane_detection_output.png"
# cv2.imwrite(output_path, final_output)
print(f"Final lane detection output saved at: {output_path}")