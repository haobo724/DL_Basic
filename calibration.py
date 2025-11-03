import numpy as np
import cv2
import glob
import tkinter as tk
from tkinter import filedialog
import os
from tkinter import messagebox

# 使用Tkinter的filedialog弹出选择文件夹的对话框

class calibration:
    def __init__(self):
        self.column_points = 0
        self.row_points = 0
        self.square_size_mm = 0
        self.pixel_to_micron_factor = 0
        
        self.cali_intermediate_results_list=[]
        
        
    def set_column_points(self, column_points):
        self.column_points = column_points
    
    def set_row_points(self, row_points):
        self.row_points = row_points
    
    def set_square_size_mm(self, square_size_mm):
        self.square_size_mm = square_size_mm
    
    def get_calib_intermediate_results(self):
        return self.cali_intermediate_results_list
    
    
    def run_cali(self):
        selected_folder = filedialog.askdirectory(title="Select folder with calibration images")

        # 检查是否选择了文件夹
        if not selected_folder:
            print("No folder selected, exiting...")
            exit()

        # 在所选文件夹下创建一个名为"Calibration_Result"的新文件夹来保存校准结果
        result_folder = os.path.join(selected_folder, "Calibration_Result")
        os.makedirs(result_folder, exist_ok=True)

        # 修改glob路径以使用用户选择的文件夹
        images = glob.glob(os.path.join(selected_folder, 'Cali*.jpg'))

        # 输入参数：棋盘格的列数和行数，以及方格的物理尺寸（毫米）
        # self.column_points = int(input("Enter column points: "))  # 将输入转换为整数
        # self.row_points = int(input("Enter row points: "))  # 将输入转换为整数
        # self.square_size_mm = float(input("Enter checkerboard size in mm: "))  # 方格的物理尺寸，保持为浮点数
        if(self.column_points == 0 or self.row_points == 0 or self.square_size_mm == 0):
            messagebox.showinfo("warning", "Please set the column points, row points and square size first")
            return
        

        # Termination criteria for the iterative algorithm
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (7,4,0)
        objp = np.zeros((self.column_points * self.row_points, 3), np.float32)  # Change to 6x8 for a 7x9 square board
        objp[:, :2] = np.mgrid[0:self.row_points, 0:self.column_points].T.reshape(-1, 2)

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        # Make sure the path is correct for your system
        # images = glob.glob('C:/Users/Lixiaoyu/Desktop/CV report/Calibration_0222/Cali*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.row_points, self.column_points), None)  # Adjusted for 8x6 inner corners

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                # Refine the corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (self.row_points, self.column_points), corners2, ret)
                self.cali_intermediate_results_list.append(img)
                # cv2.imshow('img', img)
                # cv2.waitKey(100)

        # cv2.destroyAllWindows()

        # Perform calibration only if there were any chessboards found
        if objpoints and imgpoints:
            ret, self.mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # Calculating the re-projection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            total_error = mean_error / len(objpoints)
            print(f"Total re-projection error: {total_error}")

            # Print the camera calibration values
            print("Camera matrix:", self.mtx)
            print("Distortion coefficients:", dist)
            print("Rotation Vectors:", rvecs)
            print("Translation Vectors:", tvecs)

            # Print focal length and calibration target details
            fx = self.mtx[0, 0]
            fy = self.mtx[1, 1]
            cx = self.mtx[0, 2]
            cy = self.mtx[1, 2]
            print(f"Focal Length (fx, fy): {fx}, {fy}")
            print("Calibration Target Details:")
            print("- Chessboard Size: {} x {}".format(self.row_points, self.column_points))
            # print("- Square Size: 1 unit (assuming the squares are unit squares)")

            # 在校准过程结束后添加转换因子的代码

            # 假设每个棋盘格方格的实际物理尺寸为2mm
            #square_size_mm = 2
            square_size_microns = self.square_size_mm * 1000  # 转换为微米

            # 计算棋盘格方格的平均像素大小
            # 这里假设所有图像都有相似的尺寸和放大率
            if len(imgpoints) > 0:
                # 选择第一个图像的角点来计算
                corners = imgpoints[0]
                # 计算相邻角点之间的平均距离
                distances = []
                for i in range(corners.shape[0] - 1):
                    for j in range(i + 1, corners.shape[0]):
                        dist = np.linalg.norm(corners[i] - corners[j])
                        distances.append(dist)
                avg_pixel_distance = np.mean(distances) / np.sqrt(2)  # 对角线距离转换为边长
                self.pixel_to_micron_factor = square_size_microns / avg_pixel_distance
                print(f"Pixel to Micron Conversion Factor: {self.pixel_to_micron_factor}")
            else:
                print("No image points found, cannot calculate conversion factor.")


        else:
            print("Chessboard corners were not found in any of the images.")


        # 假设我们的转换因子变量名为pixel_to_micron_factor
        np.save('pixel_to_micron_factor.npy', self.pixel_to_micron_factor)
        # After calibration
        #np.savez('calibration_data.npz', camera_matrix=mtx, dist_coeffs=dist)

#作为程序入口时才会运行下面实例化过程
if __name__ == "__main__":
    calib =calibration()
    calib.run_cali()