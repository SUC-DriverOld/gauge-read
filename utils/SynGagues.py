import numpy as np
import cv2
import math
import random


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

    xmin = 50
    ymin = 50
    zmin = 50
    xmax = 200
    ymax = 200
    zmax = 200
    red = random.random() < p_red
    if red:
        xmin = 0
        ymin = 0
        zmin = 127
        xmax = 100
        ymax = 100
        zmax = 255
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


def draw_line(img, source, dest, colour, thickness, arrow=False, arrow_scale=None, tip_length=None, shadow=False, rand=False):
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


def gen_gauge(use_homography=True, use_artefacts=False):
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
    # canvas
    H = 512
    W = 512
    h = 400  # 减小内部画布尺寸,为透视变换留出空间
    w = h
    use_border = True
    canvas_background_colour = rand_colour(p_gray=0.2)

    # gauge shape - 仪表盘强制使用圆形
    use_rectangle_gauge = False  # 强制圆形
    gauge_center_coordinates = (h // 2, w // 2)
    gauge_border_thickness = intminmax(2, 40)
    gauge_radius = min(h, w) // 2 - gauge_border_thickness // 2 - 1
    gauge_background_colour = rand_colour(p_light=0.8) if truefalse(0.7) else rand_colour(p_gray=0.2)
    gauge_border_colour = rand_colour(p_gray=0)

    # 仪表盘角度范围 - 不是360度,通常是180-270度的扇形
    gauge_type = random.choice(["半圆", "3/4圆", "大半圆"])
    if gauge_type == "半圆":
        angle_start = minmax(-135, -90)  # 起始角度
        angle_range = minmax(160, 200)  # 角度范围
    elif gauge_type == "3/4圆":
        angle_start = minmax(-135, -100)
        angle_range = minmax(250, 280)
    else:  # 大半圆
        angle_start = minmax(-150, -120)
        angle_range = minmax(210, 240)

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

    # 主刻度数量 (通常5-10个)
    num_major_ticks = intminmax(5, 11)

    # numerals - 仪表盘数值标签
    use_numerals = True  # 强制显示数字
    use_roman = False  # 仪表盘不用罗马数字
    num_font = intminmax(0, 7)
    num_font_scale = minmax(0.4, 0.9)
    num_font_thickness = intminmax(1, 3)
    num_colour = tick_colour if truefalse(0.8) else rand_colour(p_dark=0.5, p_gray=0.2)
    num_gap = intminmax(15, 35)

    # 量程值 - 仪表盘的起始和结束值
    start_value = random.choice([0, 0, 0, 10, 20, -50, -100])  # 偏向0起始
    value_ranges = [10, 20, 50, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000, 5000, 10000]
    end_value = start_value + random.choice(value_ranges)

    # 指针 - 仪表盘只有一个指针
    pointer_ratio = minmax(0.1, 0.9)  # 指针在量程中的比例
    pointer_angle = angle_start + pointer_ratio * angle_range

    # 计算当前读数
    reading_value = start_value + pointer_ratio * (end_value - start_value)

    # 指针样式
    pointer_scale = minmax(0.65, 0.85)
    pointer_back_scale = 0 if truefalse(0.5) else minmax(-0.2, 0)
    pointer_colour = rand_colour(p_red=0.3, p_dark=0.6)
    pointer_thickness = intminmax(3, 12)

    # circle - 中心圆
    use_circle_border = truefalse(0.6)
    circle_radius = intminmax(5, 10)
    circle_colour = rand_colour(p_dark=0.4)
    circle_border_colour = rand_colour(p_dark=0.1)
    circle_border_thickness = intminmax(1, 3)
    pointer_shadow = False

    # create background
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = canvas_background_colour

    # create gauge - 创建仪表盘背景
    if not use_rectangle_gauge:
        img = cv2.circle(img, gauge_center_coordinates, gauge_radius, gauge_background_colour, cv2.FILLED)
        img = cv2.circle(img, gauge_center_coordinates, gauge_radius, gauge_border_colour, gauge_border_thickness)
    else:
        img = cv2.rectangle(img, (0, 0), (h, w), gauge_background_colour, cv2.FILLED)
        img = cv2.rectangle(img, (0, 0), (h, w), gauge_border_colour, gauge_border_thickness)

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
    if use_numerals:
        num_texts = []
        if use_roman:
            # 罗马数字标签
            roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            num_texts = roman_numerals[:num_major_ticks]
        else:
            # 根据量程生成数值标签
            num_texts = [
                str(int(start_value + i * (end_value - start_value) / (num_major_ticks - 1))) for i in range(num_major_ticks)
            ]

        # 计算数值标签位置(沿着主刻度角度)
        acos2 = np.cos(major_tick_angles * math.pi / 180)
        asin2 = np.sin(major_tick_angles * math.pi / 180)

        tx = np.rint(cx + (r - gauge_border_thickness - tick_h_gap - tick_h_length - num_gap) * acos2).astype(int)
        ty = np.rint(cy + (r - gauge_border_thickness - tick_h_gap - tick_h_length - num_gap) * asin2).astype(int)

        for i in range(len(num_texts)):
            textsize = cv2.getTextSize(str(num_texts[i]), num_font, num_font_scale, num_font_thickness)[0]
            textX = tx[i] - textsize[0] // 2
            textY = ty[i] + textsize[1] // 2
            cv2.putText(img, str(num_texts[i]), (textX, textY), num_font, num_font_scale, num_colour, num_font_thickness)

    # pointer - 绘制指针(单个指针)
    source, dest = get_coordinates(cx, cy, r, pointer_scale, pointer_back_scale, pointer_angle)
    img = draw_line(img, source, dest, pointer_colour, pointer_thickness, shadow=pointer_shadow)

    # circle - 中心圆
    img = cv2.circle(img, gauge_center_coordinates, circle_radius, circle_colour, cv2.FILLED)
    if use_circle_border:
        img = cv2.circle(img, gauge_center_coordinates, circle_radius, circle_border_colour, circle_border_thickness)

    if use_border:
        IMG = np.zeros((H, W, 3), np.uint8)
        IMG[:] = canvas_background_colour
        Iy = (H - h) // 2
        Ix = (W - w) // 2
        IMG[Iy : Iy + h, Ix : Ix + w, :] = img
        img = IMG

    if use_homography:
        points = np.array(((Ix, Iy), (Ix + w, Iy), (Ix, Iy + h), (Ix + w, Iy + h)), dtype=np.float32)
        # 限制最大扰动,确保圆形不会被裁剪
        max_purturb = min(Ix, Iy) - 10  # 留出安全边距
        purturb = np.random.randint(-max_purturb, max_purturb + 1, (4, 2)).astype(np.float32)
        points2 = points + purturb
        M = cv2.getPerspectiveTransform(points, points2)
        img = cv2.warpPerspective(img, M, (H, W), borderValue=canvas_background_colour)
        Minv = cv2.findHomography(points2 * 2 / 448 - 1, points * 2 / 448 - 1)[0]

    else:
        Minv = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).astype(np.float32)

    return img, angle_start, angle_end, pointer_angle, start_value, end_value, reading_value, Minv


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path("datas/train")
    output_dir.mkdir(exist_ok=True)

    for i in range(5000):
        # 生成仪表盘
        img, angle_start, angle_end, pointer_angle, start_value, end_value, reading_value, Minv = gen_gauge()

        output_path = output_dir / f"test_gauge_{i}.png"
        cv2.imwrite(str(output_path), img)
