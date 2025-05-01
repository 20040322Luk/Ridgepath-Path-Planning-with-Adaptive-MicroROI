import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_startpoints import FindPeaks as Fp
import os

Strips = 10
Img_H = 330
Img_W = 640

def load_and_preprocess_image(image_path, output_dir):
    """
    加载图像并裁剪底部 10%，返回裁剪后的图像。
    """
    # 读取灰度图像
    source_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        raise FileNotFoundError(f"无法找到图像文件：{image_path}")

    # 调整图像大小（可选）
    # source_img = cv2.resize(source_img, (640, 480))  # 调整为固定大小
    height, width = source_img.shape

    # 裁剪图像底部 10%
    crop_end = int(height * 0.9)  # 只保留前90%的高度
    img_cropped = source_img[0:crop_end, :]

    # 保存裁剪后的图像
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/cropped_image.png', img_cropped)

    return source_img, img_cropped, crop_end

def extract_label_points(img_cropped, starting_points, num_strips=10, initial_roi_width=160, shrink_rate=0.95):
    """
    使用起始点提取标签点，返回每个起始点的标签中心点，并绘制微型 ROI 区域和标签点。
    """
    strip_height = img_cropped.shape[0] // num_strips  # 每个条带的高度
    all_label_centers = []
    img_with_rois_and_points = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)  # 将图像转换为 BGR 格式以便绘制

    for start_point in starting_points:
        label_centers = []  # 当前起始点的标签中心点列表
        start_x, start_y = start_point
        roi_width = initial_roi_width  # 初始化微型 ROI 的宽度

        # 遍历每个条带（从底部向上）
        for strip_idx in range(num_strips):
            # 计算条带的起始和结束行（从底部向上）
            strip_start = img_cropped.shape[0] - (strip_idx + 1) * strip_height
            strip_end = img_cropped.shape[0] - strip_idx * strip_height

            # 提取条带图像
            strip_img = img_cropped[strip_start:strip_end, :]

            # 定义微型 ROI 的初始位置
            roi_x_start = max(0, start_x - roi_width // 2)
            roi_x_end = min(strip_img.shape[1], start_x + roi_width // 2)

            # 提取微型 ROI 的图像
            roi_img = strip_img[:, roi_x_start:roi_x_end]

            # 找到微型 ROI 中的白色像素
            white_pixels = np.column_stack(np.where(roi_img > 0))

            if len(white_pixels) > 0:
                # 计算白色像素的中心
                center_x = roi_x_start + int(np.mean(white_pixels[:, 1]))
                center_y = strip_start + int(np.mean(white_pixels[:, 0]))
                label_centers.append((center_x, center_y))

                # 绘制 ROI 区域
                cv2.rectangle(img_with_rois_and_points, (roi_x_start, strip_start), (roi_x_end, strip_end), (0, 255, 0), 2)
                # 绘制标签点
                cv2.circle(img_with_rois_and_points, (center_x, center_y), 5, (0, 0, 255), -1)

                # 更新微型 ROI 的宽度和起始点
                roi_width = int(roi_width * shrink_rate)
                start_x = center_x
            else:
                # 如果没有找到白色像素，则跳出循环
                break

        all_label_centers.append(label_centers)

    # 保存绘制了 ROI 区域和标签点的图像

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/cropped_image_with_rois_and_points.png', img_with_rois_and_points)

    return all_label_centers

# def fit_centerline(all_label_centers, img_cropped, output_dir):
#     """
#     对所有起始点的标签中心点拟合中心线，并绘制结果。
#     """
#     img_with_centerline = cv2.imread(f'{output_dir}/cropped_image_with_rois_and_points.png')
#     # img_with_centerline = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)
#
#     for i, label_centers in enumerate(all_label_centers):
#         if len(label_centers) < 2:  # 至少需要两个点才能拟合
#             print(f"标签点不足，无法拟合第 {i+1} 组中心线。")
#             continue
#
#         # 转换为 numpy 数组
#         label_centers = np.array(label_centers)
#
#         # 拟合二次多项式
#         poly_coeffs = np.polyfit(label_centers[:, 1], label_centers[:, 0], deg=2)
#         poly_func = np.poly1d(poly_coeffs)
#
#         # 在裁剪图像上绘制拟合的中心线
#         for y in range(img_cropped.shape[0]):
#             x = int(poly_func(y))
#             if 0 <= x < img_cropped.shape[1]:
#                 img_with_centerline[y, x] = [0, 255, 0]  # 绘制绿色线条
#
#         # 绘制标签点
#         for center in label_centers:
#             cv2.circle(img_with_centerline, (center[0], center[1]), 5, (0, 0, 255), -1)
#
#     # 保存绘制了中心线和标签点的图像
#     cv2.imwrite(f'{output_dir}/cropped_image_with_centerline_and_points.png', img_with_centerline)
#
#     # 可视化拟合中心线
#     # plt.figure(figsize=(10, 6))
#     # plt.imshow(img_cropped, cmap='gray')
#
#     for i, label_centers in enumerate(all_label_centers):
#         if len(label_centers) < 2:
#             continue
#
#         label_centers = np.array(label_centers)
#         y_vals = np.arange(img_cropped.shape[0])
#         x_vals = poly_func(y_vals)
#         plt.plot(x_vals, y_vals, color='green', label=f'Fitted Centerline {i+1}')
#
#     for i, label_centers in enumerate(all_label_centers):
#         if len(label_centers) < 2:
#             continue
#
#         label_centers = np.array(label_centers)
#         plt.scatter(label_centers[:, 0], label_centers[:, 1], color='red', label='Label Centers')
#
#     # plt.legend()
#     # plt.title('Fitted Centerlines')
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     # plt.gca()
#     # plt.savefig(f'{output_dir}/fitted_centerline_visualization.png')
#     # plt.show()
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


def fit_centerline(all_label_centers, img_cropped, output_dir):
    """
    对所有起始点的标签中心点拟合中心线，并绘制结果。
    """
    img_with_centerline_path = f'{output_dir}/cropped_image_with_rois_and_points.png'
    if not os.path.exists(img_with_centerline_path):
        print("Error: 找不到带有ROI和标签点的图像文件。")
        return

    img_with_centerline = cv2.imread(img_with_centerline_path)

    for i, label_centers in enumerate(all_label_centers):
        if len(label_centers) < 2:  # 至少需要两个点才能拟合直线
            print(f"标签点不足，无法拟合第 {i + 1} 组直线。")
            continue

        # 转换为 numpy 数组
        label_centers = np.array(label_centers)

        # 分离 X 和 Y 坐标
        x_coords = label_centers[:, 0]
        y_coords = label_centers[:, 1]

        # 使用最小二乘法拟合直线
        poly_coeffs = np.polyfit(y_coords, x_coords, deg=1)  # 一阶多项式拟合
        poly_func = np.poly1d(poly_coeffs)  # 转换为多项式函数

        # 打印拟合的直线方程
        print(f"第 {i + 1} 组拟合直线方程: x = {poly_coeffs[0]:.4f} * y + {poly_coeffs[1]:.4f}")

        # 计算直线的起点和终点
        y_start = 0  # 图像顶部
        y_end = img_cropped.shape[0]  # 图像底部
        x_start = int(poly_func(y_start))  # 直线在顶部的 x 坐标
        x_end = int(poly_func(y_end))  # 直线在底部的 x 坐标

        # 确保起点和终点在图像范围内
        x_start = max(0, min(x_start, img_cropped.shape[1] - 1))
        x_end = max(0, min(x_end, img_cropped.shape[1] - 1))

        # 绘制拟合的直线
        cv2.line(img_with_centerline, (x_start, y_start), (x_end, y_end), (0, 200, 255), thickness=2)

        # 绘制标签点
        for center in label_centers:
            cv2.circle(img_with_centerline, (center[0], center[1]), 5, (0, 0, 255), -1)  # 红色点

    midpoints = calculate_midpoints(all_label_centers)
    if len(midpoints) >= 2:
        # 将中点转换为 NumPy 数组
        midpoints = np.array(midpoints)

        # 分离 X 和 Y 坐标
        x_coords = midpoints[:, 0]
        y_coords = midpoints[:, 1]

        # 使用最小二乘法拟合中线
        poly_coeffs_mid = np.polyfit(y_coords, x_coords, deg=1)  # 一阶多项式拟合
        poly_func_mid = np.poly1d(poly_coeffs_mid)  # 转换为多项式函数

        # 打印拟合的中线方程
        print(f"拟合中线方程: x = {poly_coeffs_mid[0]:.4f} * y + {poly_coeffs_mid[1]:.4f}")

        # 计算中线的起点和终点
        y_start = 0  # 图像顶部
        y_end = img_cropped.shape[0]  # 图像底部
        x_start_mid = int(poly_func_mid(y_start))  # 中线在顶部的 x 坐标
        x_end_mid = int(poly_func_mid(y_end))  # 中线在底部的 x 坐标

        # 确保中线起点和终点在图像范围内
        x_start_mid = max(0, min(x_start_mid, img_cropped.shape[1] - 1))
        x_end_mid = max(0, min(x_end_mid, img_cropped.shape[1] - 1))

        # 绘制拟合的中线
        cv2.line(img_with_centerline, (x_start_mid, y_start), (x_end_mid, y_end), (0, 0, 255), thickness=2)

        # 绘制中点
        for midpoint in midpoints:
            cv2.circle(img_with_centerline, (midpoint[0], midpoint[1]), 5, (0, 120, 180), -1)

    # 保存绘制了拟合直线和中线的图像
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/fitness_straight_line_with_midline.png', img_with_centerline)
    # 保存绘制了拟合直线和标签点的图像
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/Final_Out.png', img_with_centerline)


def fit_and_evaluate(all_label_centers):
    ssr_list = []
    for i, label_centers in enumerate(all_label_centers):
        if len(label_centers) < 2:
            print(f"标签点不足，无法拟合第 {i + 1} 组直线。")
            continue

        label_centers = np.array(label_centers)
        x_coords = label_centers[:, 0]
        y_coords = label_centers[:, 1]

        # 拟合直线
        poly_coeffs = np.polyfit(y_coords, x_coords, deg=1)
        poly_func = np.poly1d(poly_coeffs)

        # 计算拟合值
        x_fitted = poly_func(y_coords)

        # 计算残差平方和
        residuals = x_coords - x_fitted
        ssr = np.sum(residuals ** 2)
        ssr_list.append(ssr)

        print(f"第 {i + 1} 组直线的残差平方和: {ssr}")

    return ssr_list



def process(image_path, starting_points, output_dir='./outputs_without_adaptive'):
    """
    主处理流程：加载图像、提取标签点、拟合中心线。
    """
    # 加载和预处理图像
    source_img, img_cropped, crop_end = load_and_preprocess_image(image_path, output_dir)

    # 提取标签点
    all_label_centers = extract_label_points(img_cropped, starting_points)
    print(f"提取了 {len(all_label_centers)} 组标签点。")

    # 拟合中心线
    fit_centerline(all_label_centers, img_cropped, output_dir)
    ssr = fit_and_evaluate(all_label_centers)

# 使用示例
if __name__ == '__main__':
    # 输入图像路径
    image_path = 'sources/binary_img.png'
    peaks = Fp(image_path)
    # 假设起始点（可以通过其他算法或手动指定）
    start_points_pre = peaks.findpeaks_process()

    # 输出目录

    output_dir = './outputs_without_adaptive'
    # 计算起始点
    start_points = [(start_points_pre[0][0], Img_H - (int)(Img_H / Strips / 2)),
                    (start_points_pre[1][0], Img_H - (int)(Img_H / Strips / 2))]
    print(start_points)

    # 执行处理流程
    process(image_path, start_points, output_dir)