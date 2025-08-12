import cv2
import mediapipe as mp
import numpy as np
import time
import math
from math import sin, cos, tan, log, sqrt, pi, e

# --- Constants ---
CAM_WIDTH, CAM_HEIGHT = 640, 480        # Camera resolution
CALC_WIDTH, CALC_HEIGHT = 640, 480      # Calculator UI resolution
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 80   # Button dimensions
PADDING = 10
DISPLAY_HEIGHT = 80
BUTTON_START_Y = 100

MAX_NUM_HANDS = 1
MIN_PINCH_DISTANCE = 30                 # Minimum distance between thumb & index to trigger click
CLICK_COOLDOWN = 0.1                     # Delay between consecutive clicks (seconds)

# Colors (BGR format)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
GREEN = (0, 150, 0)      # Display bar
BLUE = (255, 0, 0)       # Button click highlight

# --- Global State Variables ---
current_input = ""
last_click_time = 0.0
scientific_mode = False
clicked_button_label = None
clicked_button_time = 0.0

# --- Button Layouts ---
BASIC_BUTTONS = [
    ['CLR', 'DEL', 'SCI', ''],
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['0', '.', '=', '+']
]

SCIENTIFIC_BUTTONS = [
    ['sin', 'cos', 'tan', 'sqrt'],
    ['log', '(', ')', '^'],
    ['pi', 'e', '', '']
]

def initialize_camera_and_mediapipe():
    """Initialize webcam and MediaPipe hand detection."""
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(max_num_hands=MAX_NUM_HANDS)
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    return cap, hands_detector, mp_hands

def draw_calculator_ui():
    """Render calculator interface (display + buttons)."""
    global clicked_button_label, clicked_button_time
    img_display = np.zeros((CALC_HEIGHT, CALC_WIDTH, 3), dtype=np.uint8)

    # Draw display bar
    cv2.rectangle(img_display, (PADDING, PADDING), (CALC_WIDTH - PADDING, DISPLAY_HEIGHT), GREEN, -1)
    cv2.putText(img_display, current_input[-15:], (PADDING + 20, DISPLAY_HEIGHT - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)

    # Select button set (basic or scientific)
    buttons_to_draw = SCIENTIFIC_BUTTONS + BASIC_BUTTONS if scientific_mode else BASIC_BUTTONS
    total_button_width = 4 * BUTTON_WIDTH
    start_x = (CALC_WIDTH - total_button_width) // 2

    # Draw all buttons
    for i, row in enumerate(buttons_to_draw):
        for j, label in enumerate(row):
            if label == "":
                continue
            x = start_x + j * BUTTON_WIDTH
            y = i * BUTTON_HEIGHT + BUTTON_START_Y

            button_color = WHITE
            if clicked_button_label == label and (time.time() - clicked_button_time < 0.15):
                button_color = BLUE

            cv2.rectangle(img_display, (x, y),
                          (x + BUTTON_WIDTH - PADDING, y + BUTTON_HEIGHT - PADDING),
                          button_color, -1)
            cv2.putText(img_display, label, (x + 20, y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 3)
    return img_display

def get_button_from_coordinates(x, y):
    """Determine which button (if any) is clicked based on coordinates."""
    buttons_layout = SCIENTIFIC_BUTTONS + BASIC_BUTTONS if scientific_mode else BASIC_BUTTONS
    total_button_width = 4 * BUTTON_WIDTH
    start_x = (CALC_WIDTH - total_button_width) // 2

    for i, row in enumerate(buttons_layout):
        for j, label in enumerate(row):
            if label == "":
                continue
            bx = start_x + j * BUTTON_WIDTH
            by = i * BUTTON_HEIGHT + BUTTON_START_Y
            if bx < x < bx + BUTTON_WIDTH - PADDING and by < y < by + BUTTON_HEIGHT - PADDING:
                return label
    return None

def calculate_distance(p1, p2):
    """Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def safe_evaluate_expression(expr):
    """Safely evaluate a math expression using Python's math module."""
    expr = expr.replace('^', '**')
    expr = expr.replace('pi', str(math.pi))
    expr = expr.replace('e', str(math.e))
    if not expr:
        return ""
    if expr.count('(') != expr.count(')'):
        return "Syntax Error"
    try:
        result = eval(expr, {"__builtins__": None}, math.__dict__)
        return str(result)
    except Exception as e:
        return "Error"

def handle_button_click(label):
    """Process logic when a button is clicked."""
    global current_input, scientific_mode, clicked_button_label, clicked_button_time
    clicked_button_label = label
    clicked_button_time = time.time()

    if label == "CLR":
        current_input = ""
    elif label == "DEL":
        current_input = current_input[:-1]
    elif label == "=":
        current_input = safe_evaluate_expression(current_input)
    elif label == "SCI":
        scientific_mode = not scientific_mode
    else:
        if current_input and (label in ['+', '-', '*', '/', '^', '.']) and \
                (current_input[-1] in ['+', '-', '*', '/', '^', '.']):
            if label != '.':
                current_input = current_input[:-1] + label
            else:
                if current_input.split()[-1].count('.') == 0:
                    current_input += label
        else:
            current_input += label

def main():
    """Main loop: capture camera feed, detect hand gestures, and update UI."""
    global last_click_time
    cap, hands_detector, mp_hands = initialize_camera_and_mediapipe()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        calc_img = draw_calculator_ui()

        # Hand gesture detection & button click handling
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = hand_landmarks.landmark
                x1, y1 = int(lm_list[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * CALC_WIDTH), \
                         int(lm_list[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * CALC_HEIGHT)
                x2, y2 = int(lm_list[mp_hands.HandLandmark.THUMB_TIP].x * CALC_WIDTH), \
                         int(lm_list[mp_hands.HandLandmark.THUMB_TIP].y * CALC_HEIGHT)

                # Draw pointer at midpoint
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(calc_img, (cx, cy), 10, GREEN, cv2.FILLED)

                # Trigger button click on pinch
                if calculate_distance((x1, y1), (x2, y2)) < MIN_PINCH_DISTANCE:
                    if time.time() - last_click_time > CLICK_COOLDOWN:
                        clicked_label = get_button_from_coordinates(cx, cy)
                        if clicked_label:
                            handle_button_click(clicked_label)
                            last_click_time = time.time()

        # Combine camera feed and calculator side-by-side
        combined = np.hstack((img, calc_img))
        cv2.imshow("Virtual Calculator", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
