import cv2
import numpy as np
import os


def get_saliency_modality(img):
    """
    生成针对黑色指针的显著性模态 (Saliency Modality)
    使用形态学黑帽变换 (Black Hat) 提取亮背景下的暗细节（黑色指针/刻度）
    """
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 定义结构元素。核大小决定了能提取多粗的线条。
    # 假设指针宽度适中，15x15 的核通常能很好地捕捉指针和文字，同时忽略大面积背景阴影
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # 黑帽运算：闭运算 - 原图
    # 专门用于提取比邻域更暗的图像区域（如白色表盘上的黑色指针）
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 增强可见性：归一化拉伸对比度，让提取出的特征更亮
    s_enhanced = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX)

    return s_enhanced.astype(np.uint8)


def main():
    image_dir = r"datas\test"
    output_dir = r"datas"

    # 获取目录下的第一张 jpg 图片
    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        return

    files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
    if not files:
        print(f"在 {image_dir} 中没有找到图片")
        return

    image_path = os.path.join(image_dir, files[0])
    print(f"正在处理图片: {image_path}")

    # 读取图片
    original = cv2.imread(image_path)
    if original is None:
        print("图片读取失败")
        return

    # 2. 生成显著性模态
    saliency = get_saliency_modality(original)

    # --- 可视化拼接 ---

    # 将单通道图转回 3 通道 BGR 以便拼接
    saliency_bgr = cv2.cvtColor(saliency, cv2.COLOR_GRAY2BGR)

    # 缩放以保持一致性 (这里不做 Resize，假设原图大小一致)
    h, w = original.shape[:2]

    # 添加文字标签
    cv2.putText(original, "Original (RGB)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(saliency_bgr, "BlackHat", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 水平拼接
    combined = np.hstack([original, saliency_bgr])

    # 如果图片太大，缩放显示
    view_scale = 1.0
    if combined.shape[1] > 1800:
        view_scale = 1800 / combined.shape[1]
        new_w = int(combined.shape[1] * view_scale)
        new_h = int(combined.shape[0] * view_scale)
        show_img = cv2.resize(combined, (new_w, new_h))
    else:
        show_img = combined

    print("显示结果窗口 (按任意键关闭)...")
    cv2.imshow("Multi-modal Preview", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_path = os.path.join(output_dir, "modalities_preview.jpg")
    cv2.imwrite(output_path, combined)
    print(f"对比图已保存至: {output_path}")


if __name__ == "__main__":
    main()
