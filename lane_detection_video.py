import cv2
import numpy as np

# ----------------------------------------
# Helper Functions
# ----------------------------------------

def region_of_interest(image):
    """
    Applies a polygonal mask to focus on the region where lanes are expected.
    Wider and taller triangle for more robust lane detection.
    """
    height, width = image.shape
    mask = np.zeros_like(image)

    # Define ROI polygon: bottom-left, bottom-right, top-center
    polygon = np.array([[ 
        (int(0.03 * width), height),         # Bottom-left
        (int(0.97 * width), height),         # Bottom-right
        (int(0.5 * width), int(0.35 * height)) # Top-center
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, mask

def make_coordinates(image, line_parameters):
    """
    Converts slope and intercept into pixel coordinates for drawing.
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]                # Bottom of the frame
    y2 = int(y1 * 0.6)                 # Slightly above the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Calculates average slope and intercept for left and right lanes.
    Returns pixel coordinates for lane lines.
    """
    left_fit, right_fit = [], []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue  # Avoid division by zero
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
    """
    Draws lane lines on a blank image.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 200, 200), 8)  # Dark yellow
    return line_image

def fill_lane_region(image, lines, lane_status):
    """
    Fills the lane area between left and right lines with color.
    Current lane → orange
    Opposite lane → red
    """
    lane_overlay = np.zeros_like(image)
    height, width = image.shape[:2]

    if len(lines) == 2:
        left_line, right_line = lines[0], lines[1]

        # Choose color based on lane status
        color = (0, 140, 255) if lane_status == "Current Lane" else (0, 0, 255)

        # Polygon for lane area
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], np.int32)
        cv2.fillPoly(lane_overlay, [pts], color)

    return lane_overlay

# ----------------------------------------
# Video Lane Detection with Lane Status Overlay
# ----------------------------------------

# Load video
cap = cv2.VideoCapture('lane 0.mp4')  # Replace with your video path

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('lane_detected.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (frame_width, frame_height))

# Previous frame lane memory
prev_lines = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------------
    # Preprocessing
    # ------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges, mask = region_of_interest(edges)

    # ------------------------
    # Detect lane lines
    # ------------------------
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    averaged_lines = average_slope_intercept(frame, lines)

    # ------------------------
    # Use previous frame lines if detection fails
    # ------------------------
    try:
        if len(averaged_lines) < 2 and prev_lines is not None:
            averaged_lines = prev_lines
    except:
        pass

    if len(averaged_lines) == 2:
        prev_lines = averaged_lines

    # ------------------------
    # Determine lane status (current vs opposite)
    # ------------------------
    cx = frame.shape[1] // 20
    lane_status = "Unknown"
    try:
        left_line = averaged_lines[0]
        if left_line[0] > cx:
            lane_status = "Current Lane"
        else:
            lane_status = "Opposite Lane"
    except:
        lane_status = "Unknown"

    # ------------------------
    # Visualization
    # ------------------------
    line_image = display_lines(frame, averaged_lines)
    lane_area = fill_lane_region(frame, averaged_lines, lane_status)

    combined = cv2.addWeighted(frame, 0.7, lane_area, 0.3, 0)
    final_output = cv2.addWeighted(combined, 0.9, line_image, 1, 1)

    # ---------- Beautiful Semi-Transparent Overlay ----------
    overlay = final_output.copy()
    box_color = (50, 50, 50)  # Dark grey rectangle behind text
    alpha = 0.6
    cv2.rectangle(overlay, (20, 20), (400, 80), box_color, -1)
    final_output = cv2.addWeighted(overlay, alpha, final_output, 1 - alpha, 0)

    # Overlay lane status text
    font = cv2.FONT_HERSHEY_SIMPLEX
    if lane_status == "Current Lane":
        text_color = (0, 180, 255)  # Orange
    else:
        text_color = (0, 0, 255)    # Red
    cv2.putText(final_output, f"Lane Status: {lane_status}",
                (40, 60), font, 0.8, text_color, 2, cv2.LINE_AA)

    # ------------------------
    # Display and save video
    # ------------------------
    cv2.imshow("Lane Detection", final_output)
    out.write(final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------------------
# Release resources
# ----------------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
