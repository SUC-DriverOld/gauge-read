import math
import random
import cv2
import numpy as np
from pathlib import Path


# Units text file behavior.
UNITS_FILE_NAME = "units.txt"
FALLBACK_UNITS = ["Unit"]

# Shared random ranges.
COLOR_VALUE_MIN = 0
COLOR_VALUE_MAX = 255
LIGHT_COLOR_MIN = 200
DARK_COLOR_MAX = 100

# General RGB color space for non-gray colors.
BASE_RGB_MIN = (50, 50, 50)
BASE_RGB_MAX = (200, 200, 200)

# Red-biased color range.
RED_RGB_MIN = (0, 0, 127)
RED_RGB_MAX = (100, 100, 255)

# Gauge start value type and weights.
START_TYPE_NAMES = ["zero", "pos_small", "pos_large", "neg_small", "neg_large", "decimal"]
START_TYPE_WEIGHTS = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]

# Start value candidates per type.
START_VALUES = {
    "zero": [0],
    "pos_small": [1, 2, 3, 4, 5, 6, 8, 10],
    "pos_large": [15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500],
    "neg_small": [-1, -2, -3, -4, -5, -10],
    "neg_large": [-15, -20, -25, -30, -40, -50, -60, -80, -100],
    "decimal": [0.1, 0.2, 0.5],
}

# End value candidates when start type is zero.
ZERO_RANGE_TYPE_NAMES = ["decimal", "small", "medium", "large"]
ZERO_RANGE_TYPE_WEIGHTS = [0.1, 0.3, 0.3, 0.3]
ZERO_END_VALUES = {
    "decimal": [0.6, 1.0, 1.6, 2.5, 4.0, 6.0],
    "small": [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 25],
    "medium": [30, 40, 50, 60, 80, 100, 120, 150, 160, 200],
    "large": [300, 400, 500, 600, 800, 1000, 1600, 2000, 3000, 4000, 5000],
}

# Positive/negative range extension candidates.
POS_SMALL_SPAN_CHOICES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
POS_LARGE_SPAN_CHOICES = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# Negative range branching thresholds.
NEG_SMALL_BRANCH_THRESHOLD = (0.4, 0.7)
NEG_LARGE_BRANCH_THRESHOLD = (0.4, 0.6)

# Negative-to-positive fallback choices.
NEG_SMALL_POSITIVE_END_CHOICES = [1, 2, 3, 4, 5, 10, 15, 20]
NEG_LARGE_POSITIVE_END_CHOICES = [10, 20, 30, 40, 50, 100]
NEG_SMALL_POSITIVE_MIN_STEP = 10
NEG_LARGE_POSITIVE_MIN_STEP = 50

# Decimal range extension.
DECIMAL_SPAN_CHOICES = [0.5, 1.0, 1.5, 2.0, 3.0]

# Tick number selection.
MAJOR_TICK_MIN = 4
MAJOR_TICK_MAX = 12
FALLBACK_TICK_COUNT_RANGE = (5, 10)
NICE_MANTISSAS = [1, 1.2, 1.25, 1.5, 1.6, 2, 2.5, 3, 4, 5, 6, 8]

# Main output and internal drawing sizes.
OUTPUT_SIZE = (512, 512)
INNER_CANVAS_SIZE = (400, 400)

# Gauge geometry.
GAUGE_BORDER_THICKNESS_RANGE = (2, 40)
ANGLE_START_RANGE = (45, 270)
ANGLE_SWEEP_RANGE = (150, 320)

# Tick styling.
USE_MINOR_TICK_PROB = 0.7
MINOR_TICK_COUNT_RANGE = (20, 60)
TICK_GAP_RANGE = (5, 20)
TICK_LENGTH_RANGE = (8, 15)
TICK_THICKNESS_RANGE = (1, 3)

# Major tick styling.
MAJOR_TICK_EXTRA_LENGTH_RANGE = (5, 25)
MAJOR_TICK_THICKNESS_RANGE = (3, 8)
KEEP_MAJOR_TICK_COLOR_PROB = 0.8

# Number text style.
FONT_INDEX_RANGE = (0, 7)
LARGE_FONT_PROB = 0.8
LARGE_FONT_SCALE_RANGE = (0.6, 1.0)
SMALL_FONT_SCALE_RANGE = (0.4, 0.6)
SMALL_FONT_THIN_STROKE_PROB = 0.8
NUMBER_COLOR_KEEP_TICK_PROB = 0.8
NUMBER_GAP_RANGE = (15, 35)

# Unit text style.
UNIT_RADIUS_RATIO_RANGE = (0.3, 0.5)
UNIT_ANGLE_JITTER_RANGE = (-10, 10)
UNIT_SCALE_FACTOR_RANGE = (0.6, 0.9)
FANCY_FONT_INDEXES = [6, 7]
UNIT_FALLBACK_FONTS = [0, 2, 3, 4]
SMALL_FONT_SCALE_THRESHOLD = 0.5
UNIT_MIN_STROKE_FOR_LARGE_FONT = 2

# Pointer behavior.
POINTER_RATIO_RANGE = (0.0, 1.0)
POINTER_SCALE_RANGE = (0.65, 0.85)
POINTER_BACK_SCALE_PROB_ZERO = 0.5
POINTER_BACK_SCALE_RANGE = (-0.2, 0)
POINTER_THICKNESS_RANGE = (2, 8)

# Center circle behavior.
USE_CENTER_BORDER_PROB = 0.6
CENTER_RADIUS_RANGE = (5, 10)
CENTER_BORDER_THICKNESS_RANGE = (1, 3)

# Artefact effects.
POINTER_SHADOW_PROB = 0.5
ARTEFACT_LINE_COUNT_RANGE = (0, 5)
RANDOM_LINE_THICKNESS_RANGE = (1, 8)

# Perspective transform perturbation safety margin.
PERSPECTIVE_SAFE_MARGIN = 5

# Post-process augmentations.
RGB_SCALE_RANGE = (0.9, 1.1)
BLUR_PROB = 0.5
BLUR_KERNEL_RANGE = (1, 10)
NOISE_PROB = 0.5
NOISE_AMPLITUDE = 10.0
RESIZE_PROB = 0.5
RESIZE_RANGE = (64, 256)

# Probabilities for grayscale/light/dark/red color branches.
CANVAS_BG_GRAY_PROB = 0.2
GAUGE_BG_LIGHT_PROB = 0.7
GAUGE_BG_LIGHT_COLOR_PROB = 0.8
TICK_COLOR_DARK_PROB = 0.5
TICK_COLOR_GRAY_PROB = 0.2
MAJOR_TICK_COLOR_DARK_PROB = 0.5
MAJOR_TICK_COLOR_GRAY_PROB = 0.2
NUMBER_COLOR_DARK_PROB = 0.5
NUMBER_COLOR_GRAY_PROB = 0.2
POINTER_COLOR_RED_PROB = 0.3
POINTER_COLOR_DARK_PROB = 0.6
CENTER_COLOR_DARK_PROB = 0.4
CENTER_BORDER_COLOR_DARK_PROB = 0.1
SHADOW_ALT_DARK_PROB = 0.8


def load_units():
    """Load display units from local text file."""
    path = Path(__file__).parent / UNITS_FILE_NAME
    if not path.exists():
        return FALLBACK_UNITS
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


COMMON_UNITS = load_units()


def get_random_unit():
    return random.choice(COMMON_UNITS)


def truefalse(p):
    return random.random() < p


def intminmax(mi, ma):
    return random.choice(range(mi, ma + 1))


def minmax(mi, ma):
    return np.random.uniform(mi, ma)


def rand_colour(p_gray=0, p_light=0, p_dark=0, p_red=0):
    xmin = COLOR_VALUE_MIN
    xmax = COLOR_VALUE_MAX
    gray = random.random() < p_gray
    if gray:
        x = random.choice(range(xmin, xmax))
        return (x, x, x)

    light = random.random() < p_light
    if light:
        xmin = LIGHT_COLOR_MIN
        x = random.choice(range(xmin, xmax))
        return (x, x, x)

    dark = random.random() < p_dark
    if dark:
        xmax = DARK_COLOR_MAX
        x = random.choice(range(xmin, xmax))
        return (x, x, x)

    xmin, ymin, zmin = BASE_RGB_MIN
    xmax, ymax, zmax = BASE_RGB_MAX
    red = random.random() < p_red
    if red:
        xmin, ymin, zmin = RED_RGB_MIN
        xmax, ymax, zmax = RED_RGB_MAX

    x = random.choice(range(xmin, xmax))
    y = random.choice(range(ymin, ymax))
    z = random.choice(range(zmin, zmax))
    return (x, y, z)


def get_coordinates(cx, cy, r, scale, back_scale, a):
    x1 = (cx + scale * r * np.cos(a * math.pi / 180)).astype(int)
    y1 = (cy + scale * r * np.sin(a * math.pi / 180)).astype(int)
    x2 = (cx + back_scale * r * np.cos(a * math.pi / 180)).astype(int)
    y2 = (cy + back_scale * r * np.sin(a * math.pi / 180)).astype(int)
    return (x1, y1), (x2, y2)


def draw_line(img, source, dest, colour, thickness, arrow=False, arrow_scale=None, tip_length=None, shadow=False):
    img = cv2.line(img, source, dest, colour, thickness)
    x1, y1 = source
    x2, y2 = dest

    if arrow:
        arrowhead = (int(x2 + (x1 - x2) * arrow_scale), int(y2 + (y1 - y2) * arrow_scale))
        img = cv2.arrowedLine(img, dest, arrowhead, colour, thickness, tipLength=tip_length)

    if shadow:
        dx = random.choice([1, -1]) * intminmax(thickness, 30)
        dy = random.choice([1, -1]) * intminmax(thickness, 30)
        shadow_colour = colour if truefalse(0.5) else rand_colour(p_dark=SHADOW_ALT_DARK_PROB)
        shadow_alpha = minmax(0.1, 0.9)
        img_orig = img.copy()
        img_shadow = cv2.line(img, (x1 + dx, y1 + dy), (x2 + dx, y2 + dy), shadow_colour, thickness)
        img = cv2.addWeighted(img_shadow, shadow_alpha, img_orig, 1 - shadow_alpha, 0)
    return img


def draw_rotated_text(img, text, center, angle, font, scale, color, thickness):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_w, text_h = text_size

    # Keep a square patch so rotation never clips text.
    diag = int(math.sqrt(text_w**2 + text_h**2)) + 4
    patch_w = diag
    patch_h = diag

    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cx, cy = patch_w // 2, patch_h // 2

    origin_x = cx - text_w // 2
    origin_y = cy + text_h // 2
    cv2.putText(mask, text, (origin_x, origin_y), font, scale, 255, thickness)

    # Rotate so text baseline is aligned with dial direction.
    rot_angle = -(angle + 90)
    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, M, (patch_w, patch_h))

    x, y = center
    tl_x = int(x - cx)
    tl_y = int(y - cy)

    img_h, img_w = img.shape[:2]
    p_x1 = max(0, tl_x)
    p_y1 = max(0, tl_y)
    p_x2 = min(img_w, tl_x + patch_w)
    p_y2 = min(img_h, tl_y + patch_h)

    if p_x1 >= p_x2 or p_y1 >= p_y2:
        return img

    m_x1 = p_x1 - tl_x
    m_y1 = p_y1 - tl_y
    m_x2 = m_x1 + (p_x2 - p_x1)
    m_y2 = m_y1 + (p_y2 - p_y1)

    roi = img[p_y1:p_y2, p_x1:p_x2]
    mask_patch = rotated_mask[m_y1:m_y2, m_x1:m_x2]

    color_patch = np.zeros_like(roi)
    color_patch[:] = color

    _, bin_mask = cv2.threshold(mask_patch, 10, 255, cv2.THRESH_BINARY)
    bin_mask_inv = cv2.bitwise_not(bin_mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=bin_mask_inv)
    img_fg = cv2.bitwise_and(color_patch, color_patch, mask=bin_mask)
    dst = cv2.add(img_bg, img_fg)

    img[p_y1:p_y2, p_x1:p_x2] = dst
    return img


def draw_unit(img, unit_text, center, radius, start_angle, end_angle, font, scale, color, thickness):
    cx, cy = center

    # Place unit text near dial center opening.
    mid_angle = (start_angle + end_angle) / 2
    mid_angle = mid_angle % 360
    unit_radius_ratio = minmax(*UNIT_RADIUS_RATIO_RANGE)
    unit_radius = radius * unit_radius_ratio

    pos_angle = mid_angle + 180
    pos_angle += minmax(*UNIT_ANGLE_JITTER_RANGE)

    rad = math.radians(pos_angle)
    tx = int(cx + unit_radius * math.cos(rad))
    ty = int(cy + unit_radius * math.sin(rad))

    if font in FANCY_FONT_INDEXES:
        font = random.choice(UNIT_FALLBACK_FONTS)

    if scale < SMALL_FONT_SCALE_THRESHOLD:
        thickness = 1
    else:
        thickness = max(thickness, UNIT_MIN_STROKE_FOR_LARGE_FONT)

    img = draw_rotated_text(img, unit_text, (tx, ty), mid_angle, font, scale, color, thickness)
    return img


def get_gauge_config():
    # 1) Sample start value type.
    start_type = random.choices(START_TYPE_NAMES, weights=START_TYPE_WEIGHTS, k=1)[0]
    start_val = random.choice(START_VALUES[start_type])

    # 2) Sample end value according to start type.
    end_val = 100

    if start_type == "zero":
        range_type = random.choices(ZERO_RANGE_TYPE_NAMES, weights=ZERO_RANGE_TYPE_WEIGHTS, k=1)[0]
        end_val = random.choice(ZERO_END_VALUES[range_type])

    elif start_type == "pos_small":
        end_val = start_val + random.choice(POS_SMALL_SPAN_CHOICES)

    elif start_type == "pos_large":
        end_val = start_val + random.choice(POS_LARGE_SPAN_CHOICES)

    elif start_type == "neg_small":
        dice = random.random()
        if dice < NEG_SMALL_BRANCH_THRESHOLD[0]:
            end_val = abs(start_val)
        elif dice < NEG_SMALL_BRANCH_THRESHOLD[1]:
            end_val = 0
        else:
            end_val = random.choice(NEG_SMALL_POSITIVE_END_CHOICES)
            if end_val <= start_val:
                end_val = start_val + NEG_SMALL_POSITIVE_MIN_STEP

    elif start_type == "neg_large":
        dice = random.random()
        if dice < NEG_LARGE_BRANCH_THRESHOLD[0]:
            end_val = abs(start_val)
        elif dice < NEG_LARGE_BRANCH_THRESHOLD[1]:
            end_val = 0
        else:
            end_val = random.choice(NEG_LARGE_POSITIVE_END_CHOICES)
            if end_val <= start_val:
                end_val = start_val + NEG_LARGE_POSITIVE_MIN_STEP

    elif start_type == "decimal":
        end_val = start_val + random.choice(DECIMAL_SPAN_CHOICES)

    # 3) Pick major tick count using preferred mantissa heuristics.
    span = end_val - start_val
    candidates = []

    for n in range(MAJOR_TICK_MIN, MAJOR_TICK_MAX + 1):
        step = span / (n - 1)
        if step <= 0:
            continue

        exponent = math.floor(math.log10(step))
        mantissa = step / (10**exponent)

        for nm in NICE_MANTISSAS:
            if abs(mantissa - nm) < 1e-4:
                candidates.append(n)
                break

    if not candidates:
        num_ticks = random.randint(*FALLBACK_TICK_COUNT_RANGE)
    else:
        num_ticks = random.choice(candidates)

    step = span / (num_ticks - 1)
    is_decimal = not (abs(step - round(step)) < 1e-6 and abs(start_val - round(start_val)) < 1e-6)

    return start_val, end_val, num_ticks, is_decimal


def draw_random_lines(img, cx, cy, r, R, num=3):
    for _ in range(num):
        r1 = intminmax(r, R)
        r2 = intminmax(r, R)
        theta1 = minmax(0, 360)
        theta2 = minmax(0, 360)
        colour = rand_colour()
        thickness = intminmax(*RANDOM_LINE_THICKNESS_RANGE)

        x1 = (cx + r1 * np.cos(theta1 * math.pi / 180)).astype(int)
        y1 = (cy + r1 * np.sin(theta1 * math.pi / 180)).astype(int)
        x2 = (cx + r2 * np.cos(theta2 * math.pi / 180)).astype(int)
        y2 = (cy + r2 * np.sin(theta2 * math.pi / 180)).astype(int)

        img = cv2.line(img, (x1, y1), (x2, y2), colour, thickness)
    return img


def gen_gauge(use_homography=True, use_artefacts=False, use_arguments=False):
    # Canvas sizes.
    H, W = OUTPUT_SIZE
    h, w = INNER_CANVAS_SIZE

    # Base canvas and outer style.
    canvas_background_colour = rand_colour(p_gray=CANVAS_BG_GRAY_PROB)
    gauge_center_coordinates = (h // 2, w // 2)
    gauge_border_thickness = intminmax(*GAUGE_BORDER_THICKNESS_RANGE)
    gauge_radius = min(h, w) // 2 - gauge_border_thickness // 2 - 1
    gauge_background_colour = (
        rand_colour(p_light=GAUGE_BG_LIGHT_COLOR_PROB)
        if truefalse(GAUGE_BG_LIGHT_PROB)
        else rand_colour(p_gray=CANVAS_BG_GRAY_PROB)
    )
    gauge_border_colour = rand_colour(p_gray=0)

    # Angle layout.
    angle_start = minmax(*ANGLE_START_RANGE)
    angle_range = minmax(*ANGLE_SWEEP_RANGE)
    angle_end = angle_start + angle_range

    # Minor tick style.
    use_minor_tick = truefalse(USE_MINOR_TICK_PROB)
    tick_gap = intminmax(*TICK_GAP_RANGE)
    tick_length = intminmax(*TICK_LENGTH_RANGE)
    tick_thickness = intminmax(*TICK_THICKNESS_RANGE)
    tick_colour = rand_colour(p_dark=TICK_COLOR_DARK_PROB, p_gray=TICK_COLOR_GRAY_PROB)

    # Major tick style.
    tick_h_gap = tick_gap
    tick_h_length = intminmax(tick_length + MAJOR_TICK_EXTRA_LENGTH_RANGE[0], MAJOR_TICK_EXTRA_LENGTH_RANGE[1])
    tick_h_thickness = intminmax(*MAJOR_TICK_THICKNESS_RANGE)
    tick_h_colour = (
        tick_colour
        if truefalse(KEEP_MAJOR_TICK_COLOR_PROB)
        else rand_colour(p_dark=MAJOR_TICK_COLOR_DARK_PROB, p_gray=MAJOR_TICK_COLOR_GRAY_PROB)
    )

    # Number text style.
    num_font = intminmax(*FONT_INDEX_RANGE)
    if truefalse(LARGE_FONT_PROB):
        num_font_scale = minmax(*LARGE_FONT_SCALE_RANGE)
    else:
        num_font_scale = minmax(*SMALL_FONT_SCALE_RANGE)

    if num_font_scale < 0.55:
        num_font_thickness = 1 if truefalse(SMALL_FONT_THIN_STROKE_PROB) else 2
    else:
        num_font_thickness = intminmax(1, 3)

    num_colour = (
        tick_colour
        if truefalse(NUMBER_COLOR_KEEP_TICK_PROB)
        else rand_colour(p_dark=NUMBER_COLOR_DARK_PROB, p_gray=NUMBER_COLOR_GRAY_PROB)
    )
    num_gap = intminmax(*NUMBER_GAP_RANGE)

    # Value range and pointer reading.
    start_value, end_value, num_major_ticks, is_decimal = get_gauge_config()
    pointer_ratio = minmax(*POINTER_RATIO_RANGE)
    angle_pointer = angle_start + pointer_ratio * angle_range
    reading_value = start_value + pointer_ratio * (end_value - start_value)

    # Pointer style.
    pointer_scale = minmax(*POINTER_SCALE_RANGE)
    pointer_back_scale = 0 if truefalse(POINTER_BACK_SCALE_PROB_ZERO) else minmax(*POINTER_BACK_SCALE_RANGE)
    pointer_colour = rand_colour(p_red=POINTER_COLOR_RED_PROB, p_dark=POINTER_COLOR_DARK_PROB)
    pointer_thickness = intminmax(*POINTER_THICKNESS_RANGE)

    # Center circle style.
    use_circle_border = truefalse(USE_CENTER_BORDER_PROB)
    circle_radius = intminmax(*CENTER_RADIUS_RANGE)
    circle_colour = rand_colour(p_dark=CENTER_COLOR_DARK_PROB)
    circle_border_colour = rand_colour(p_dark=CENTER_BORDER_COLOR_DARK_PROB)
    circle_border_thickness = intminmax(*CENTER_BORDER_THICKNESS_RANGE)
    pointer_shadow = False

    if use_artefacts:
        pointer_shadow = truefalse(POINTER_SHADOW_PROB)
        num_random_lines = intminmax(*ARTEFACT_LINE_COUNT_RANGE)

    # Create base image.
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = canvas_background_colour

    # Draw gauge face.
    img = cv2.circle(img, gauge_center_coordinates, gauge_radius, gauge_background_colour, cv2.FILLED)
    img = cv2.circle(img, gauge_center_coordinates, gauge_radius, gauge_border_colour, gauge_border_thickness)

    cy, cx = gauge_center_coordinates
    r = gauge_radius
    num_ticks = intminmax(*MINOR_TICK_COUNT_RANGE)
    tick_angles = np.linspace(angle_start, angle_end, num_ticks)
    major_tick_angles = np.linspace(angle_start, angle_end, num_major_ticks)

    # Draw minor ticks.
    if use_minor_tick:
        for angle in tick_angles:
            rad = angle * math.pi / 180
            x1 = int(cx + (r - gauge_border_thickness - tick_gap) * np.cos(rad))
            y1 = int(cy + (r - gauge_border_thickness - tick_gap) * np.sin(rad))
            x2 = int(cx + (r - gauge_border_thickness - tick_gap - tick_length) * np.cos(rad))
            y2 = int(cy + (r - gauge_border_thickness - tick_gap - tick_length) * np.sin(rad))
            img = cv2.line(img, (x1, y1), (x2, y2), tick_colour, tick_thickness)

    # Draw major ticks.
    for angle in major_tick_angles:
        rad = angle * math.pi / 180
        h_x1 = int(cx + (r - gauge_border_thickness - tick_h_gap) * np.cos(rad))
        h_y1 = int(cy + (r - gauge_border_thickness - tick_h_gap) * np.sin(rad))
        h_x2 = int(cx + (r - gauge_border_thickness - tick_h_gap - tick_h_length) * np.cos(rad))
        h_y2 = int(cy + (r - gauge_border_thickness - tick_h_gap - tick_h_length) * np.sin(rad))
        img = cv2.line(img, (h_x1, h_y1), (h_x2, h_y2), tick_h_colour, tick_h_thickness)

    # Draw number labels.
    num_texts = []
    for i in range(num_major_ticks):
        val = start_value + i * (end_value - start_value) / (num_major_ticks - 1)
        if is_decimal:
            step = (end_value - start_value) / (num_major_ticks - 1)
            text = f"{val:.2f}" if step < 0.1 else f"{val:.1f}"
        else:
            text = str(int(round(val)))
        num_texts.append(text)

    acos2 = np.cos(major_tick_angles * math.pi / 180)
    asin2 = np.sin(major_tick_angles * math.pi / 180)
    tx = np.rint(cx + (r - gauge_border_thickness - tick_h_gap - tick_h_length - num_gap) * acos2).astype(int)
    ty = np.rint(cy + (r - gauge_border_thickness - tick_h_gap - tick_h_length - num_gap) * asin2).astype(int)

    for i in range(len(num_texts)):
        img = draw_rotated_text(
            img=img,
            text=str(num_texts[i]),
            center=(tx[i], ty[i]),
            angle=major_tick_angles[i],
            font=num_font,
            scale=num_font_scale,
            color=num_colour,
            thickness=num_font_thickness,
        )

    # Draw unit label.
    unit_text = get_random_unit()
    unit_scale = num_font_scale * minmax(*UNIT_SCALE_FACTOR_RANGE)
    img = draw_unit(img, unit_text, (cx, cy), r, angle_start, angle_end, num_font, unit_scale, num_colour, num_font_thickness)

    # Draw pointer.
    source, dest = get_coordinates(cx, cy, r, pointer_scale, pointer_back_scale, angle_pointer)
    img = draw_line(img, source, dest, pointer_colour, pointer_thickness, shadow=pointer_shadow)

    if use_artefacts:
        img = draw_random_lines(img, cx, cy, circle_radius, r, num=num_random_lines)

    # Draw center circle.
    img = cv2.circle(img, gauge_center_coordinates, circle_radius, circle_colour, cv2.FILLED)
    if use_circle_border:
        img = cv2.circle(img, gauge_center_coordinates, circle_radius, circle_border_colour, circle_border_thickness)

    # Put inner canvas into output canvas.
    IMG = np.zeros((H, W, 3), np.uint8)
    IMG[:] = canvas_background_colour
    Iy = (H - h) // 2
    Ix = (W - w) // 2
    IMG[Iy : Iy + h, Ix : Ix + w, :] = img
    img = IMG

    # Optional perspective warp.
    center_pt = np.array([[[Ix + w / 2.0, Iy + h / 2.0]]], dtype=np.float32)
    if use_homography:
        points = np.array(((Ix, Iy), (Ix + w, Iy), (Ix, Iy + h), (Ix + w, Iy + h)), dtype=np.float32)
        max_purturb = min(Ix, Iy) - PERSPECTIVE_SAFE_MARGIN
        purturb = np.random.randint(-max_purturb, max_purturb + 1, (4, 2)).astype(np.float32)
        points2 = points + purturb
        M = cv2.getPerspectiveTransform(points, points2)
        img = cv2.warpPerspective(img, M, (H, W), borderValue=canvas_background_colour)
        Minv = cv2.findHomography(points2 * 2 / 448 - 1, points * 2 / 448 - 1)[0]
        center_pt_warped = cv2.perspectiveTransform(center_pt, M)
        final_cx, final_cy = center_pt_warped[0][0]
    else:
        Minv = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).astype(np.float32)
        final_cx, final_cy = center_pt[0][0]

    # Optional image augmentations.
    if use_arguments:
        r = img[:, :, 0] * np.random.uniform(*RGB_SCALE_RANGE)
        g = img[:, :, 1] * np.random.uniform(*RGB_SCALE_RANGE)
        b = img[:, :, 2] * np.random.uniform(*RGB_SCALE_RANGE)
        rgb = [r, g, b]

        img = np.stack(rgb, 2)
        if truefalse(BLUR_PROB):
            k = np.random.randint(*BLUR_KERNEL_RANGE)
            img = cv2.blur(img, (k, k))
        if truefalse(NOISE_PROB):
            img_h, img_w, _ = np.shape(img)
            img = img + NOISE_AMPLITUDE * np.random.uniform(-1.0, 1.0, (img_h, img_w, 3))
        if truefalse(RESIZE_PROB):
            sz = np.random.randint(*RESIZE_RANGE)
            img_h, img_w, _ = np.shape(img)
            img = cv2.resize(img, (sz, sz))
            final_cx = final_cx * sz / img_w
            final_cy = final_cy * sz / img_h

    return (img, (angle_start, angle_end, angle_pointer), (start_value, end_value, reading_value), Minv, (final_cx, final_cy))


if __name__ == "__main__":
    imgs = []
    for _ in range(20):
        img, angle, value, Minv, center = gen_gauge()
        img = cv2.resize(img, OUTPUT_SIZE)
        imgs.append(img)

    rows, cols = 4, 5
    h, w = imgs[0].shape[:2]

    row_imgs = []
    for r in range(rows):
        row = np.hstack(imgs[r * cols : (r + 1) * cols])
        row_imgs.append(row)
    final_img = np.vstack(row_imgs)

    output_path = Path("datas/gauge_grid.png")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), final_img)
    print(f"Saved gauge grid image to {output_path}")
