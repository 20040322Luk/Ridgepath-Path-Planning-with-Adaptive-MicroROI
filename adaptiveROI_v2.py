import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_startpoints import FindPeaks as Fp
import os
import csv

Strips = 10 # 预设分割条带数
Img_H = 330 # 输入图像的高度
Img_W = 640 # 输入图像的宽度
Roi_son_width = 50 # 子ROI的宽度

def load_and_preprocess_image(image_path, output_dir):
    """
    加载图像并裁剪底部 10%，返回裁剪后的图像。
    """
    source_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        raise FileNotFoundError(f"无法找到图像文件：{image_path}")

    height, width = source_img.shape

    # 裁剪图像底部 10%
    crop_end = int(height * 0.9)  # 只保留前90%的高度
    img_cropped = source_img[0:crop_end, :]

    # 保存裁剪后的图像
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/cropped_image.png', img_cropped)

    return source_img, img_cropped, crop_end

def extract_label_points(img_cropped, starting_points, num_strips=10, initial_roi_width=Roi_son_width, height_factor=1.0, i=1, T=0.05):
    """
    使用自适应ROI算法提取标签点，返回每个起始点的标签中心点，并绘制微型 ROI 区域和标签点。
    同时在图像中绘制条带分界线。
    """
    strip_height = img_cropped.shape[0] // num_strips  # 每个条带的高度
    all_label_centers = []
    img_with_rois_and_points = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)  # 将图像转换为 BGR 格式以便绘制

    # 绘制条带分界线
    for strip_idx in range(num_strips + 1):
        y = img_cropped.shape[0] - strip_idx * strip_height
        cv2.line(img_with_rois_and_points, (0, y), (img_cropped.shape[1], y), (255, 255, 0), 1)  # 黄色分界线

    for start_point in starting_points:
        label_centers = []  # 当前起始点的标签中心点列表
        start_x, start_y = start_point

        # 遍历每个条带（从底部向上）
        for strip_idx in range(num_strips):
            # 计算条带的起始和结束行（从底部向上）
            strip_start = img_cropped.shape[0] - (strip_idx + 1) * strip_height
            strip_end = img_cropped.shape[0] - strip_idx * strip_height

            # 提取条带图像
            strip_img = img_cropped[strip_start:strip_end, :]

            # 定义子ROI的高度
            hrl = strip_height
            hrr = strip_height

            # 初始化子ROI的中心
            cl = start_x
            cr = start_x
            ml = 0
            mr = 0

            # 搜索左子ROI
            for x in range(1, 17):
                mask_l = generate_mask(strip_img, cl, initial_roi_width, hrl)
                res_l = cv2.bitwise_and(strip_img, mask_l)
                white_percentage_l = calculate_white_percentage(res_l)

                if white_percentage_l < T:
                    ml = abs(start_x - cl)
                    break

                cl -= x * i
                if cl < 0 or is_overlapping(cl, initial_roi_width, hrl, all_label_centers):
                    ml = abs(start_x - cl)
                    break

            # 搜索右子ROI
            for x in range(1, 17):
                mask_r = generate_mask(strip_img, cr, initial_roi_width, hrr)
                res_r = cv2.bitwise_and(strip_img, mask_r)
                white_percentage_r = calculate_white_percentage(res_r)

                if white_percentage_r < T:
                    mr = abs(start_x - cr)
                    break

                cr += x * i
                if cr >= strip_img.shape[1] or is_overlapping(cr, initial_roi_width, hrr, all_label_centers):
                    mr = abs(start_x - cr)
                    break

            # 更新标签中心
            if ml > 0 and mr > 0:
                new_center_x = (cl + cr) // 2
                new_center_y = strip_start + strip_height // 2
                label_centers.append((new_center_x, new_center_y))

                # 绘制 ROI 区域
                cv2.rectangle(img_with_rois_and_points, (cl, strip_start), (cl - initial_roi_width, strip_end), (255, 0, 0), 2)
                cv2.rectangle(img_with_rois_and_points, (cr, strip_start), (cr + initial_roi_width, strip_end), (0, 255, 255), 2)
                # 绘制标签点
                cv2.circle(img_with_rois_and_points, (new_center_x, new_center_y), 5, (0, 0, 255), -1)

                # 更新起始点
                start_x = new_center_x
            else:
                # 如果没有找到白色像素，则跳出循环
                break

        all_label_centers.append(label_centers)

    # 保存绘制了 ROI 区域、标签点和条带分界线的图像
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/rois_and_points.png', img_with_rois_and_points)

    return all_label_centers


def calculate_midpoints(all_label_centers):
    """
    计算每条条带中所有标签点的中点。
    """
    num_strips = max([len(label_centers) for label_centers in all_label_centers])
    midpoints = []
    for strip_idx in range(num_strips):
        points_in_strip = [label_centers[strip_idx] for label_centers in all_label_centers if
                           strip_idx < len(label_centers)]
        if points_in_strip:
            avg_x = int(np.mean([point[0] for point in points_in_strip]))
            avg_y = int(np.mean([point[1] for point in points_in_strip]))
            midpoints.append((avg_x, avg_y))
    return midpoints

def fit_and_draw_lines(all_label_centers, img_cropped, output_dir='./outputs', line_thickness=2, poly_degree=4):
    """
    对所有标签点进行多项式曲线拟合，并绘制拟合结果。
    同时生成一个仅包含标签点和中点的图像，并绘制条带分界线。
    """
    # 创建绘制拟合曲线的图像
    img_with_lines = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)

    # 创建仅绘制标签点和中点的图像
    img_mid_points = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)

    # 绘制条带分界线
    strip_height = img_cropped.shape[0] // Strips
    for strip_idx in range(Strips + 1):
        y = img_cropped.shape[0] - strip_idx * strip_height
        cv2.line(img_with_lines, (0, y), (img_cropped.shape[1], y), (255, 255, 0), 1)  # 黄色分界线
        cv2.line(img_mid_points, (0, y), (img_cropped.shape[1], y), (255, 255, 0), 1)  # 黄色分界线

    # 遍历每组标签点
    for i, label_centers in enumerate(all_label_centers):
        if len(label_centers) < 2:  # 至少需要两个点才能拟合曲线
            print(f"标签点不足，无法拟合第 {i + 1} 组曲线。")
            continue

        # 将标签点转换为 NumPy 数组
        label_centers = np.array(label_centers)

        # 分离 X 和 Y 坐标
        x_coords = label_centers[:, 0]
        y_coords = label_centers[:, 1]

        # 使用多项式拟合曲线
        poly_coeffs = np.polyfit(y_coords, x_coords, deg=poly_degree)  # 二次多项式拟合
        poly_func = np.poly1d(poly_coeffs)  # 转换为多项式函数

        # 打印拟合的曲线方程
        print(f"第 {i + 1} 组拟合曲线方程: x = {poly_coeffs}")

        # 计算曲线上的点
        y_vals = np.arange(0, img_cropped.shape[0])  # 从顶部到底部的所有行
        x_vals = poly_func(y_vals).astype(int)  # 计算每一行对应的 x 坐标

        # 绘制拟合的曲线
        for y, x in zip(y_vals, x_vals):
            if 0 <= x < img_cropped.shape[1]:  # 确保点在图像范围内
                img_with_lines[y, x] = (0, 0, 0)  # 绘制绿色曲线

        # 绘制原始标签点
        for center in label_centers:
            cv2.circle(img_with_lines, (center[0], center[1]), 5, (0, 0, 255), -1)  # 红色点
            cv2.circle(img_mid_points, (center[0], center[1]), 5, (0, 0, 255), -1)  # 同时绘制在中点图像中

    # 计算中点的中线
    midpoints = calculate_midpoints(all_label_centers)
    if len(midpoints) >= 2:
        # 将中点转换为 NumPy 数组
        midpoints = np.array(midpoints)

        # 分离 X 和 Y 坐标
        x_coords = midpoints[:, 0]
        y_coords = midpoints[:, 1]

        # 使用多项式拟合中线
        poly_coeffs_mid = np.polyfit(y_coords, x_coords, deg=poly_degree)  # 二次多项式拟合
        poly_func_mid = np.poly1d(poly_coeffs_mid)  # 转换为多项式函数

        # 打印拟合的中线方程
        print(f"拟合中线方程: x = {poly_coeffs_mid}")

        # 计算中线的点
        y_vals = np.arange(0, img_cropped.shape[0])
        x_vals_mid = poly_func_mid(y_vals).astype(int)

        # 绘制拟合的中线
        for y, x in zip(y_vals, x_vals_mid):
            if 0 <= x < img_cropped.shape[1]:
                img_with_lines[y, x] = (255, 0, 0)  # 绘制蓝色中线

        # 绘制中点
        for midpoint in midpoints:
            cv2.circle(img_with_lines, (midpoint[0], midpoint[1]), 5, (0, 120, 180), -1)  # 蓝色点
            cv2.circle(img_mid_points, (midpoint[0], midpoint[1]), 5, (0, 120, 180), -1)  # 同时绘制在中点图像中

        # 采样中线点并保存为 CSV 文件
        sampled_points = list(zip(x_vals_mid, y_vals))  # 获取采样点的 (x, y) 坐标
        with open(f'{output_dir}/midline_points.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X', 'Y'])  # 写入表头
            writer.writerows(sampled_points)  # 写入采样点
    # 保存绘制了拟合曲线和中线的图像
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/fitness_curve_with_midline.png', img_with_lines)

    # 保存仅包含标签点和中点的图像
    cv2.imwrite(f'{output_dir}/MID_Point.png', img_mid_points)


def generate_mask(img, center, width, height):
    """
    生成掩码模板。
    """
    mask = np.zeros(img.shape, dtype=np.uint8)
    left = max(0, center - width // 2)
    right = min(img.shape[1], center + width // 2)
    mask[0:height, left:right] = 255
    return mask

def calculate_white_percentage(image):
    """
    计算白像素的百分比。
    """
    white_pixels = np.sum(image == 255)
    total_pixels = image.size
    return white_pixels / total_pixels

def is_overlapping(center, width, height, fitted_rois):
    """
    检查当前子ROI是否与已拟合的子ROI重叠。
    """
    for roi in fitted_rois:
        for point in roi:
            if abs(center - point[0]) < width // 2 + Roi_son_width // 2:
                return True
    return False

def process(image_path, starting_points, output_dir='./outputs/'):
    """
    主处理流程：加载图像、提取标签点、拟合中心线。
    """
    # 加载和预处理图像
    source_img, img_cropped, crop_end = load_and_preprocess_image(image_path, output_dir)

    # 提取标签点
    all_label_centers = extract_label_points(img_cropped, starting_points)
    print(f"提取了 {len(all_label_centers)} 组标签点。")

    # 拟合中心线
    fit_and_draw_lines(all_label_centers, img_cropped, output_dir)

# 使用示例
if __name__ == '__main__':
    # 输入图像路径
    image_path = 'sources/binary_img.png'
    peaks = Fp(image_path)
    # 假设起始点（可以通过其他算法或手动指定）
    start_points_pre = peaks.findpeaks_process()

    # 输出目录
    output_dir = './outputs'

    # 计算起始点
    start_points = [(start_points_pre[0][0], Img_H - (int)(Img_H / Strips / 2)),
                    (start_points_pre[1][0], Img_H - (int)(Img_H / Strips / 2))]
    print(start_points)

    # 执行处理流程
    process(image_path, start_points, output_dir)
