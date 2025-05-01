import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class FindPeaks:
    def __init__(self, image_path, output_dir='./outputs/'):
        self.image_path = image_path
        self.output_dir = output_dir
        self.source_img = None
        self.img_cropped = None
        self.img_warped = None
        self.img_warped_with_peaks = None
        self.source_img_with_peaks = None
        self.M = None
        self.peaks = None
        self.peaks_in_original = None
        self.column_sum = None
        self.height_cropped = None
        self.line_position = None
        self.output_size = (640, 640)  # 投影图像的输出尺寸

    def load_and_preprocess_image(self):
        # 读取原始图像（灰度模式）
        self.source_img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.source_img = cv2.resize(self.source_img, (640, 330))
        height, width = self.source_img.shape

        # 计算横线位置（图像底部 30%）
        self.line_position = int(height * 0.8)

        # 裁剪图像底部区域
        self.img_cropped = self.source_img[self.line_position:height, 0:width]
        self.height_cropped, width_cropped = self.img_cropped.shape

        # 定义裁剪区域的四个角点
        src_points = np.float32([
            [0, 0],
            [width_cropped - 1, 0],
            [width_cropped - 1, self.height_cropped - 1],
            [0, self.height_cropped - 1]
        ])

        # 定义目标区域的四个角点
        dst_points = np.float32([
            [0, 0],
            [self.output_size[0] - 1, 0],
            [self.output_size[0] - 1, self.output_size[1] - 1],
            [1, self.output_size[0] - 1]
        ])

        # 计算透视变换矩阵
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用透视变换
        self.img_warped = cv2.warpPerspective(self.img_cropped, self.M, self.output_size)

        # 保存投影图像
        cv2.imwrite(f'{self.output_dir}/warped_image.png', self.img_warped)

    def detect_peaks(self):
        # 检测峰值（按列求和）
        self.column_sum = np.sum(self.img_warped, axis=0)  # 沿着行方向求和
        self.peaks, _ = find_peaks(self.column_sum)  # 检测峰值

        # 标记峰值位置
        self.img_warped_with_peaks = cv2.cvtColor(self.img_warped, cv2.COLOR_GRAY2BGR)
        for peak in self.peaks:
            cv2.circle(self.img_warped_with_peaks, (peak, self.height_cropped // 2), 5, (0, 0, 255), -1)

        # 保存标记峰值的投影图像
        cv2.imwrite(f'{self.output_dir}/warped_image_with_peaks.png', self.img_warped_with_peaks)

        # 绘制峰值曲线
        plt.figure(figsize=(10, 5))
        plt.plot(self.column_sum, label='Column Sum')
        plt.plot(self.peaks, self.column_sum[self.peaks], "x", label='Peaks')
        plt.legend()
        plt.title('Column Sum and Peaks')
        plt.xlabel('Column Index')
        plt.ylabel('Pixel Sum')
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/column_sum_peaks.png')

    def inverse_warp_peaks(self):
        # 计算逆透视变换矩阵
        M_inv = cv2.invert(self.M)[1]

        # 将峰值位置从目标图像反投影到原始裁剪图像
        self.peaks_in_original = []
        for peak in self.peaks:
            # 假设峰值在目标图像的中间行
            peak_point = np.array([peak, self.height_cropped // 2, 1], dtype=np.float32)
            original_point = np.dot(M_inv, peak_point)
            original_point /= original_point[2]  # 归一化

            # 将裁剪图像的坐标转换为原始图像的坐标
            original_point[1] += self.line_position
            self.peaks_in_original.append((int(original_point[0]), int(original_point[1])))

        # 在原始图像上标记反投影后的峰值位置
        self.source_img_with_peaks = cv2.cvtColor(self.source_img, cv2.COLOR_GRAY2BGR)
        for peak in self.peaks_in_original:
            cv2.circle(self.source_img_with_peaks, peak, 5, (0, 0, 255), -1)

        # 保存结果图像
        cv2.imwrite(f'{self.output_dir}/source_image_with_peaks.png', self.source_img_with_peaks)
        # h,w = self.source_img_with_peaks.shape[:2]
        # crop_start = int(h * 0.7)
        #
        # # 裁剪图像底部 30%
        # img_cropped = self.source_img_with_peaks[0:crop_start, :]
        #

    def findpeaks_process(self):
        # 加载和预处理图像
        self.load_and_preprocess_image()

        # 检测峰值
        self.detect_peaks()

        # 反投影峰值位置
        self.inverse_warp_peaks()

        # 检查结果
        if len(self.peaks) == 0:
            print("在当前帧中未检测到峰值，增加裁剪比例进行更多探索。")
            return []  # 返回空列表表示未检测到峰值
        elif len(self.peaks_in_original) < 1:
            print("未检测到足够的峰值，当前帧被拒绝处理。")
            return []  # 返回空列表表示未检测到足够的峰值
        else:
            print(f"检测到 {len(self.peaks_in_original)} 个峰值，可以进行后续处理。")

            return self.peaks_in_original  # 返回峰值坐标列表

# 使用示例
if __name__ == '__main__':
    peaks = FindPeaks('sources/binary_img.png')
    points = peaks.findpeaks_process()
    print(points)
