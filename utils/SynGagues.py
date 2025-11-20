import numpy as np
import cv2
import math
import random
from pathlib import Path


def load_units():
    path = Path(__file__).parent / "units.txt"
    if not path.exists():
        return ["Unit"]
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
    xmin = 0
    xmax = 255
    gray = random.random() < p_gray
    if gray:
        x = random.choice(range(xmin, xmax))
        return (x, x, x)
    light = random.random() < p_light
    if light:
        xmin = 200
        x = random.choice(range(xmin, xmax))
        return (x, x, x)
    dark = random.random() < p_dark
    if dark:
        xmax = 100
        x = random.choice(range(xmin, xmax))
        return (x, x, x)

    xmin, ymin, zmin = 50, 50, 50
    xmax, ymax, zmax = 200, 200, 200
    red = random.random() < p_red
    if red:
        xmin, ymin, zmin = 0, 0, 127
        xmax, ymax, zmax = 100, 100, 255
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
        shadow_colour = colour if truefalse(0.5) else rand_colour(p_dark=0.8)
        shadow_alpha = minmax(0.1, 0.9)
        img_orig = img.copy()
        img_shadow = cv2.line(img, (x1 + dx, y1 + dy), (x2 + dx, y2 + dy), shadow_colour, thickness)
        img = cv2.addWeighted(img_shadow, shadow_alpha, img_orig, 1 - shadow_alpha, 0)
    return img


def draw_rotated_text(img, text, center, angle, font, scale, color, thickness):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_w, text_h = text_size

    # Canvas size for rotation
    diag = int(math.sqrt(text_w**2 + text_h**2)) + 4
    patch_w = diag
    patch_h = diag

    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    cx, cy = patch_w // 2, patch_h // 2

    origin_x = cx - text_w // 2
    origin_y = cy + text_h // 2
    cv2.putText(mask, text, (origin_x, origin_y), font, scale, 255, thickness)

    # Rotate mask
    # angle is the angle of the tick. We want text bottom to point to center.
    # At -90 deg (top), text is upright (0 rot). Rotation needed = angle + 90.
    # cv2 uses CCW degrees. So we rotate by -(angle + 90).
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

    ret, bin_mask = cv2.threshold(mask_patch, 10, 255, cv2.THRESH_BINARY)
    bin_mask_inv = cv2.bitwise_not(bin_mask)

    img_bg = cv2.bitwise_and(roi, roi, mask=bin_mask_inv)
    img_fg = cv2.bitwise_and(color_patch, color_patch, mask=bin_mask)
    dst = cv2.add(img_bg, img_fg)

    img[p_y1:p_y2, p_x1:p_x2] = dst
    return img

def draw_unit(img, unit_text, center, radius, start_angle, end_angle, font, scale, color, thickness):
    cx, cy = center
    
    # Calculate the "center" of the gauge arc
    mid_angle = (start_angle + end_angle) / 2
    mid_angle = mid_angle % 360
    
    # Position the unit opposite to the arc center (in the opening)
    # But slightly closer to the center than the ticks
    unit_radius_ratio = minmax(0.3, 0.5)
    unit_radius = radius * unit_radius_ratio
    
    # Position angle is opposite to mid_angle
    pos_angle = mid_angle + 180
    
    # Add small random jitter to position angle
    pos_angle += minmax(-10, 10)
    
    rad = math.radians(pos_angle)
    tx = int(cx + unit_radius * math.cos(rad))
    ty = int(cy + unit_radius * math.sin(rad))
    
    # Filter fancy fonts (6: Script Simplex, 7: Script Complex)
    if font in [6, 7]:
        font = random.choice([0, 2, 3, 4]) # Simplex, Duplex, Complex, Triplex
        
    # Ensure thickness is visible but not too thick for small scales
    if scale < 0.5:
        thickness = 1
    else:
        thickness = max(thickness, 2)
    
    # Use draw_rotated_text to align text with gauge orientation
    # Passing mid_angle ensures the text is "upright" relative to the gauge face
    img = draw_rotated_text(img, unit_text, (tx, ty), mid_angle, font, scale, color, thickness)
    return img


def get_gauge_config():
    # 1. 随机选择起始值类型
    start_type = random.choices(
        ["zero", "pos_small", "pos_large", "neg_small", "neg_large", "decimal"], weights=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1], k=1
    )[0]

    start_val = 0
    if start_type == "zero":
        start_val = 0
    elif start_type == "pos_small":
        start_val = random.choice([1, 2, 3, 4, 5, 6, 8, 10])
    elif start_type == "pos_large":
        start_val = random.choice([15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500])
    elif start_type == "neg_small":
        start_val = random.choice([-1, -2, -3, -4, -5, -10])
    elif start_type == "neg_large":
        start_val = random.choice([-15, -20, -25, -30, -40, -50, -60, -80, -100])
    elif start_type == "decimal":
        start_val = random.choice([0.1, 0.2, 0.5])

    # 2. 根据起始值选择结束值
    end_val = 100

    if start_type == "zero":
        range_type = random.choices(["decimal", "small", "medium", "large"], weights=[0.1, 0.3, 0.3, 0.3], k=1)[0]
        if range_type == "decimal":
            end_val = random.choice([0.6, 1.0, 1.6, 2.5, 4.0, 6.0])
        elif range_type == "small":
            end_val = random.choice([1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 25])
        elif range_type == "medium":
            end_val = random.choice([30, 40, 50, 60, 80, 100, 120, 150, 160, 200])
        else:
            end_val = random.choice([300, 400, 500, 600, 800, 1000, 1600, 2000, 3000, 4000, 5000])

    elif start_type == "pos_small":
        r = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
        end_val = start_val + r

    elif start_type == "pos_large":
        r = random.choice([50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
        end_val = start_val + r

    elif start_type == "neg_small":
        dice = random.random()
        if dice < 0.4:  # Symmetric
            end_val = abs(start_val)
        elif dice < 0.7:  # To Zero
            end_val = 0
        else:  # To Positive
            end_val = random.choice([1, 2, 3, 4, 5, 10, 15, 20])
            if end_val <= start_val:
                end_val = start_val + 10

    elif start_type == "neg_large":
        dice = random.random()
        if dice < 0.4:  # Symmetric
            end_val = abs(start_val)
        elif dice < 0.6:  # To Zero
            end_val = 0
        else:  # To Positive
            end_val = random.choice([10, 20, 30, 40, 50, 100])
            if end_val <= start_val:
                end_val = start_val + 50

    elif start_type == "decimal":
        r = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])
        end_val = start_val + r

    # 3. 确定刻度数量
    span = end_val - start_val
    candidates = []

    # 常见的仪表盘步长尾数
    nice_mantissas = [1, 1.2, 1.25, 1.5, 1.6, 2, 2.5, 3, 4, 5, 6, 8]

    for n in range(4, 13):
        step = span / (n - 1)
        if step <= 0:
            continue

        exponent = math.floor(math.log10(step))
        mantissa = step / (10**exponent)

        for nm in nice_mantissas:
            if abs(mantissa - nm) < 1e-4:
                candidates.append(n)
                break

    if not candidates:
        num_ticks = random.randint(5, 10)
    else:
        num_ticks = random.choice(candidates)

    step = span / (num_ticks - 1)
    if abs(step - round(step)) < 1e-6 and abs(start_val - round(start_val)) < 1e-6:
        is_decimal = False
    else:
        is_decimal = True

    return start_val, end_val, num_ticks, is_decimal


def draw_random_lines(img, cx, cy, r, R, num=3):
    for _ in range(num):
        r1 = intminmax(r, R)
        r2 = intminmax(r, R)
        theta1 = minmax(0, 360)
        theta2 = minmax(0, 360)
        colour = rand_colour()
        thickness = intminmax(1, 8)

        x1 = (cx + r1 * np.cos(theta1 * math.pi / 180)).astype(int)
        y1 = (cy + r1 * np.sin(theta1 * math.pi / 180)).astype(int)
        x2 = (cx + r2 * np.cos(theta2 * math.pi / 180)).astype(int)
        y2 = (cy + r2 * np.sin(theta2 * math.pi / 180)).astype(int)

        img = cv2.line(img, (x1, y1), (x2, y2), colour, thickness)
    return img


def gen_gauge(use_homography=True, use_artefacts=False, use_arguments=False):
    """
    生成合成仪表盘图像

    Returns:
            img: 生成的图像
            start_angle: 起始刻度角度 (0-360)
            end_angle: 结束刻度角度 (0-360)
            pointer_angle: 指针角度 (0-360)
            start_value: 起始刻度值
            end_value: 结束刻度值
            reading_value: 当前读数
            Minv: 透视变换逆矩阵
    """
    # hyperparameters:
    H = 512
    W = 512
    w = h = 400  # 减小内部画布尺寸,为透视变换留出空间
    canvas_background_colour = rand_colour(p_gray=0.2)

    # gauge shape
    gauge_center_coordinates = (h // 2, w // 2)
    gauge_border_thickness = intminmax(2, 40)
    gauge_radius = min(h, w) // 2 - gauge_border_thickness // 2 - 1
    gauge_background_colour = rand_colour(p_light=0.8) if truefalse(0.7) else rand_colour(p_gray=0.2)
    gauge_border_colour = rand_colour(p_gray=0)

    # 仪表盘角度范围
    # 起始角度: 45 (右下) 到 270 (上) 之间
    angle_start = minmax(45, 270)
    # 扫过的角度范围: 150 到 320 度
    angle_range = minmax(150, 320)
    angle_end = angle_start + angle_range

    # ticks - 仪表盘刻度
    use_minor_tick = truefalse(0.7)  # 小刻度
    tick_gap = intminmax(5, 20)
    tick_length = intminmax(8, 15)
    tick_thickness = intminmax(1, 3)
    tick_colour = rand_colour(p_dark=0.5, p_gray=0.2)

    # major ticks - 主刻度
    tick_h_gap = tick_gap
    tick_h_length = intminmax(tick_length + 5, 25)
    tick_h_thickness = intminmax(3, 8)
    tick_h_colour = tick_colour if truefalse(0.8) else rand_colour(p_dark=0.5, p_gray=0.2)

    # numerals - 仪表盘数值标签
    num_font = intminmax(0, 7)
    
    # 调整字体大小分布，减少过小字体的比例
    if truefalse(0.8):
        num_font_scale = minmax(0.6, 1.0) # 正常/大字体
    else:
        num_font_scale = minmax(0.4, 0.6) # 小字体 (20% 概率)

    # 调整字体粗细，避免小字体过粗
    if num_font_scale < 0.55:
        # 小字体时，大部分情况使用细线条
        num_font_thickness = 1 if truefalse(0.8) else 2
    else:
        num_font_thickness = intminmax(1, 3)

    num_colour = tick_colour if truefalse(0.8) else rand_colour(p_dark=0.5, p_gray=0.2)
    num_gap = intminmax(15, 35)

    # 量程值 - 仪表盘的起始和结束值
    start_value, end_value, num_major_ticks, is_decimal = get_gauge_config()

    # 指针
    pointer_ratio = minmax(0.0, 1.0)  # 指针在量程中的位置
    pointer_angle = angle_start + pointer_ratio * angle_range

    # 计算当前读数
    reading_value = start_value + pointer_ratio * (end_value - start_value)

    # 指针样式
    pointer_scale = minmax(0.65, 0.85)
    pointer_back_scale = 0 if truefalse(0.5) else minmax(-0.2, 0)
    pointer_colour = rand_colour(p_red=0.3, p_dark=0.6)
    pointer_thickness = intminmax(2, 8)

    # circle - 中心圆
    use_circle_border = truefalse(0.6)
    circle_radius = intminmax(5, 10)
    circle_colour = rand_colour(p_dark=0.4)
    circle_border_colour = rand_colour(p_dark=0.1)
    circle_border_thickness = intminmax(1, 3)
    pointer_shadow = False

    if use_artefacts:
        pointer_shadow = truefalse(0.5)
        num_random_lines = intminmax(0, 5)

    # create background
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = canvas_background_colour

    # create gauge - 创建仪表盘背景
    img = cv2.circle(img, gauge_center_coordinates, gauge_radius, gauge_background_colour, cv2.FILLED)
    img = cv2.circle(img, gauge_center_coordinates, gauge_radius, gauge_border_colour, gauge_border_thickness)

    # create ticks - 创建仪表盘刻度(扇形范围内)
    cy, cx = gauge_center_coordinates
    r = gauge_radius

    # 生成刻度角度 - 只在仪表盘的角度范围内
    num_ticks = intminmax(20, 60)  # 小刻度数量
    tick_angles = np.linspace(angle_start, angle_end, num_ticks)

    # 主刻度角度
    major_tick_angles = np.linspace(angle_start, angle_end, num_major_ticks)

    # 绘制小刻度
    if use_minor_tick:
        for angle in tick_angles:
            rad = angle * math.pi / 180
            x1 = int(cx + (r - gauge_border_thickness - tick_gap) * np.cos(rad))
            y1 = int(cy + (r - gauge_border_thickness - tick_gap) * np.sin(rad))
            x2 = int(cx + (r - gauge_border_thickness - tick_gap - tick_length) * np.cos(rad))
            y2 = int(cy + (r - gauge_border_thickness - tick_gap - tick_length) * np.sin(rad))
            img = cv2.line(img, (x1, y1), (x2, y2), tick_colour, tick_thickness)

    # 绘制主刻度
    for angle in major_tick_angles:
        rad = angle * math.pi / 180
        h_x1 = int(cx + (r - gauge_border_thickness - tick_h_gap) * np.cos(rad))
        h_y1 = int(cy + (r - gauge_border_thickness - tick_h_gap) * np.sin(rad))
        h_x2 = int(cx + (r - gauge_border_thickness - tick_h_gap - tick_h_length) * np.cos(rad))
        h_y2 = int(cy + (r - gauge_border_thickness - tick_h_gap - tick_h_length) * np.sin(rad))
        img = cv2.line(img, (h_x1, h_y1), (h_x2, h_y2), tick_h_colour, tick_h_thickness)

    # create numerals - 创建仪表盘数值标签(扇形范围内)
    num_texts = []
    # 根据量程生成数值标签
    for i in range(num_major_ticks):
        val = start_value + i * (end_value - start_value) / (num_major_ticks - 1)
        if is_decimal:
            step = (end_value - start_value) / (num_major_ticks - 1)
            if step < 0.1:
                text = f"{val:.2f}"
            else:
                text = f"{val:.1f}"
        else:
            text = str(int(round(val)))
        num_texts.append(text)

    # 计算数值标签位置(沿着主刻度角度)
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

    # Draw unit - 绘制单位
    unit_text = get_random_unit()
    unit_scale = num_font_scale * minmax(0.6, 0.9)
    img = draw_unit(img, unit_text, (cx, cy), r, angle_start, angle_end, num_font, unit_scale, num_colour, num_font_thickness)

    # pointer - 绘制指针
    source, dest = get_coordinates(cx, cy, r, pointer_scale, pointer_back_scale, pointer_angle)
    img = draw_line(img, source, dest, pointer_colour, pointer_thickness, shadow=pointer_shadow)

    if use_artefacts:
        img = draw_random_lines(img, cx, cy, circle_radius, r, num=num_random_lines)

    # circle - 中心圆
    img = cv2.circle(img, gauge_center_coordinates, circle_radius, circle_colour, cv2.FILLED)
    if use_circle_border:
        img = cv2.circle(img, gauge_center_coordinates, circle_radius, circle_border_colour, circle_border_thickness)

    IMG = np.zeros((H, W, 3), np.uint8)
    IMG[:] = canvas_background_colour
    Iy = (H - h) // 2
    Ix = (W - w) // 2
    IMG[Iy : Iy + h, Ix : Ix + w, :] = img
    img = IMG

    if use_homography:
        points = np.array(((Ix, Iy), (Ix + w, Iy), (Ix, Iy + h), (Ix + w, Iy + h)), dtype=np.float32)
        # 限制最大扰动,确保圆形不会被裁剪
        max_purturb = min(Ix, Iy) - 5  # 留出安全边距
        purturb = np.random.randint(-max_purturb, max_purturb + 1, (4, 2)).astype(np.float32)
        points2 = points + purturb
        M = cv2.getPerspectiveTransform(points, points2)
        img = cv2.warpPerspective(img, M, (H, W), borderValue=canvas_background_colour)
        Minv = cv2.findHomography(points2 * 2 / 448 - 1, points * 2 / 448 - 1)[0]
    else:
        Minv = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).astype(np.float32)

    if use_arguments:
        r = img[:,:,0] * np.random.uniform(0.9, 1.1)
        g = img[:,:,1] * np.random.uniform(0.9, 1.1)
        b = img[:,:,2] * np.random.uniform(0.9, 1.1)
        rgb = [r,g,b]

        img = np.stack(rgb, 2)
        if truefalse(0.5): # blur
            k = np.random.randint(1,10)
            img = cv2.blur(img, (k,k))
        if truefalse(0.5): # noise
            H, W, _ = np.shape(img)
            img = img + 10 * np.random.uniform(-1.0, 1.0, (H, W, 3))
        if truefalse(0.5): # resize
            sz = np.random.randint(64, 256)
            img = cv2.resize(img, (sz, sz))

    return img, angle_start, angle_end, pointer_angle, start_value, end_value, reading_value, Minv


if __name__ == "__main__":
    from pathlib import Path

    imgs = []
    for i in range(20):
        img, angle_start, angle_end, pointer_angle, start_value, end_value, reading_value, Minv = gen_gauge()
        print(
            f"Gauge {i+1}: {start_value} to {end_value}, Angle: {angle_start:.1f} to {angle_end:.1f}, Pointer Angle: {pointer_angle:.1f}, Reading: {reading_value:.1f}"
        )
        img = cv2.resize(img, (512, 512))
        imgs.append(img)

    rows = 4
    cols = 5
    h, w = imgs[0].shape[:2]

    row_imgs = []
    for r in range(rows):
        row = np.hstack(imgs[r * cols : (r + 1) * cols])
        row_imgs.append(row)
    final_img = np.vstack(row_imgs)

    output_path = Path("datas/gauge_grid.png")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), final_img)
