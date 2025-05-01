import numpy as np
import csv

def transform_pixel_to_world(pixel_points, camera_matrix, rotation_matrix, translation_vector, ground_height=0):
    """
    根据图片中的公式，实现从像素坐标到世界坐标的矩阵变换。

    参数:
        pixel_points: 像素坐标列表 [(u1, v1), (u2, v2), ...]
        camera_matrix: 相机内参矩阵 (3x3)
        rotation_matrix: 旋转矩阵 R (3x3)
        translation_vector: 平移向量 T (3x1)
        ground_height: 地面高度 Z_w，默认为 0

    返回:
        world_points: 世界坐标列表 [(Xw1, Yw1, Zw1), (Xw2, Yw2, Zw2), ...]
    """
    # 将像素坐标增加一个维度（第三维全为1）
    pixel_coords_homogeneous = np.hstack((pixel_points, np.ones((len(pixel_points), 1), dtype=np.float32)))

    # 计算内参矩阵的逆
    inverse_camera_matrix = np.linalg.inv(camera_matrix)

    # 将像素坐标转换到相机坐标系
    camera_coords = (inverse_camera_matrix @ pixel_coords_homogeneous.T).T  # 转置后批量计算

    # 打印像素坐标和相机坐标转换结果
    print(f"像素坐标 -> 相机坐标:\n{np.hstack((pixel_points, camera_coords))}")

    # 假设地面高度为 Z_w = 0，根据比例调整相机坐标
    scale_factors = ground_height / camera_coords[:, 2]  # 计算比例因子
    camera_coords *= scale_factors[:, np.newaxis]  # 按比例缩放相机坐标

    # 从相机坐标转换到世界坐标
    world_coords = (rotation_matrix @ (camera_coords - translation_vector).T).T * 100  # 旋转并转换到世界坐标系

    return world_coords


def read_pixel_points_from_csv(file_path):
    """
    从 CSV 文件中读取像素坐标。

    参数:
        file_path: CSV 文件路径

    返回:
        pixel_points: 像素坐标列表 [(x1, y1), (x2, y2), ...]
    """
    pixel_points = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行（标题行）
        for row in reader:
            x, y = float(row[0]), float(row[1])
            pixel_points.append((x, y))
    return pixel_points


def save_world_points_to_csv(file_path, world_points):
    """
    将世界坐标保存到 CSV 文件。

    参数:
        file_path: 保存的 CSV 文件路径
        world_points: 世界坐标列表 [(Xw1, Yw1, Zw1), (Xw2, Yw2), ...]
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(['Xw', 'Yw', 'Zw'])
        # 写入数据
        writer.writerows(world_points)


# 示例代码
if __name__ == '__main__':
    # 从 CSV 文件读取像素坐标
    input_csv_file_path = './outputs/midline_points.csv'  # 替换为你的输入 CSV 文件路径
    pixel_points = read_pixel_points_from_csv(input_csv_file_path)

    # 摄像头内参矩阵 (fx, fy, cx, cy)
    camera_matrix = np.array([
        [800, 0, 300],  # fx, cx
        [0, 400, 280],  # fy, cy
        [0, 0, 1]
    ], dtype=np.float32)

    # 旋转矩阵 R
    alpha = np.radians(30)  # 旋转角度 30 度
    rotation_matrix = np.array([
        [-1, 0, 0],
        [0, np.sin(alpha), np.cos(alpha)],
        [0, -np.cos(alpha), -np.sin(alpha)]
    ], dtype=np.float32)

    # 平移向量 T
    translation_vector = np.array([0, 0, 0], dtype=np.float32)  # 假设相机离地面 1 米

    # 地面高度
    ground_height = 2  # 假设地面在 Z_w = 0

    # 调用函数
    world_points = transform_pixel_to_world(pixel_points, camera_matrix, rotation_matrix, translation_vector, ground_height)

    # 保存结果到 CSV 文件
    output_csv_file_path = './outputs/world_points.csv'  # 替换为你的输出 CSV 文件路径
    save_world_points_to_csv(output_csv_file_path, world_points)

    # 输出结果到控制台
    print("像素坐标:", pixel_points)
    print("世界坐标已保存到:", output_csv_file_path)
