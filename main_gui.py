import cv2
from tkinter import END, filedialog
import tkinter as tk
from camera import Camera
from PIL import Image, ImageTk
from calibration import calibration
import numpy as np

def only_numeric_input(P):
    # 如果字符串是空的或者所有字符都是数字，则允许输入
    if P.isdigit() or P == "":
        return True
    else:
        return False

class gui_basics:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Capture") 
        self.root.geometry("1000x1000")
        self.cam = Camera()
        self.cali = calibration()
        # variable not layout elements
        self.capture_number = 0
        self.snap_imgs = []
        self.frame_np = None
        self.frame = None
        self.snap_saved_folder = ''
        self.cali_imm = None
        self.cali_imm_single = None
    
        # left-top layout
        self.Info_frame = tk.Frame(self.root, width=300, height=200, background="red")
        self.Info_frame.grid(row=0,column=0)

        self.button_grab_start = tk.Button(self.Info_frame, text='Capture',width=20, heigh=2,command=self.start_grab)
        self.button_grab_start.grid(row=0,column=0, padx=120,pady=35)
        self.button_grab_stop = tk.Button(self.Info_frame,text='Stop Capture',width=20,height=2, command=self.stop_grab)
        self.button_grab_stop.grid(row=1,column=0)
        self.button_choose_folder = tk.Button(self.Info_frame,text='Choose Snap Saved Folder',width=20,height=2, command=self.choose_snap_saved_folder)
        self.button_choose_folder.grid(row=2,column=0)
        self.snap_number = tk.Label(self.Info_frame, text="Frame number: 0") # 显示当前记录了的帧数
        self.snap_number.grid(row=0,column=1)
        self.snap_info =tk.Text(self.Info_frame, width=30, height=10)
        self.snap_info.grid(row=1,column=1)
        
        vcmd = (self.root.register(only_numeric_input), '%P')
        self.button_calibrate = tk.Button(self.Info_frame, text='Calibrate',width=20, heigh=2,command=self.sync_cali_parameters)
        self.button_calibrate.grid(row=3,column=0)
        self.cali_col =tk.Entry(self.Info_frame,validate="key", validatecommand=vcmd)
        self.cali_col.grid(row=3,column=1)
        self.cali_row =tk.Entry(self.Info_frame, validate="key", validatecommand=vcmd)
        self.cali_row.grid(row=4,column=1)
        self.cali_square_size =tk.Entry(self.Info_frame, validate="key", validatecommand=vcmd)
        self.cali_square_size.grid(row=5,column=1)
    
        self.Camera_frame = None
        self.camera_window = None
        self.cali_window = None
     
        
    def run(self):
        self.create_camera_frame(frame_handler=self.update_display)
        self.create_cali_window(cali_handler=self.wait_cali_finish)
        self.root.mainloop()
        
    def __del__(self):
        if self.cam.cap.isOpened():
            self.cam.cap.release()
            
    def create_cali_window(self,cali_handler):
        self.cali_window = tk.Label(self.Camera_frame,height=200,width=200, image=None)#一个窗口用来展示calibration的画面
        self.cali_window.grid(row=1,column=1)
        cali_handler()
        
        
    def create_camera_frame(self,frame_handler):
        self.Camera_frame = tk.Frame(self.root, width=600, height=200, background="blue")
        self.Camera_frame.grid(row=1, column=0, columnspan=8)
        self.camera_window = tk.Label(self.Camera_frame,height=200,width=200, image=None)#一个窗口用来展示现在摄像头的画面
        self.camera_window.grid(row=0,column=0)
        frame_handler()
        
    def update_cali_window(self,index=0):
        if len(self.cali_imm) > index:  # 检查是否还有更多图像要展示
            img = self.cali_imm[index]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            self.cali_imm_single = ImageTk.PhotoImage(Image.fromarray(img))
            self.cali_window.configure(image=self.cali_imm_single)
            # 使用after设置1秒延迟来调用show_imm展示下一个图像
            self.root.after(1000, lambda: self.update_cali_window(index + 1))
        else:
            # 如果没有更多的图像要展示，可能需要做一些结束处理
            print("All images have been displayed.")
        
    
    def wait_cali_finish(self):
        result_from_cali = self.cali.get_calib_intermediate_results()
        if len(result_from_cali) > 0:
            self.cali_imm=result_from_cali
            self.root.after(1, self.update_cali_window)
                
        else:
            self.root.after(1, self.wait_cali_finish)

        
    def save_snapshot(self,event):
        self.capture_number += 1
        self.snap_number.config(text=f"Frame number: {self.capture_number}")
        self.snap_imgs.append(self.frame_np)
        # print(f"Frame {self.capture_number} saved.")
        capture_img = self.frame_np
        cv2.imwrite(f"{self.snap_saved_folder}/Cali{self.capture_number:02}.jpg", capture_img)
        self.snap_number.config(text=f"Frame number: {self.capture_number}")
        self.snap_info.insert(END, f"Frame {self.capture_number} saved.\n")

    def choose_snap_saved_folder(self):
        self.snap_saved_folder = filedialog.askdirectory(title="Select Folder for Calibration Images")

    def update_display(self):
        result_from_cam = self.cam.get_camera_img()
        if isinstance(result_from_cam,np.ndarray):
            result_from_cam = cv2.cvtColor(result_from_cam, cv2.COLOR_BGR2RGB)
            self.frame_np = result_from_cam.copy()
            result_from_cam = cv2.resize(result_from_cam, (200, 200))
            
            self.frame = ImageTk.PhotoImage(Image.fromarray(result_from_cam))
            
            self.camera_window.configure(image=self.frame )
            self.root.after(1, self.update_display)
        elif (result_from_cam == -1):
            print("No frame from camera.")
        else:
            print("Camera stopped.")
            self.root.after(1, self.update_display)

    def start_grab(self):
        if self.snap_saved_folder == '':
            tk.messagebox.showinfo("Warning", "Please first choose a folder to save the images.")
            return
        self.root.bind('<space>', self.save_snapshot)
        print("Start capturing.")

        
    def stop_grab(self):
        self.root.unbind('<space>')
        print("Stop capturing.")

    def sync_cali_parameters(self):
        if self.cali_col.get() == '' or self.cali_row.get() == '' or self.cali_square_size.get() == '':
            tk.messagebox.showinfo("Warning", "Please set the column points, row points and square size first")
            return
        self.cali.set_column_points(int(self.cali_col.get()))
        self.cali.set_row_points(int(self.cali_row.get()))
        self.cali.set_square_size_mm(float(self.cali_square_size.get()))
        self.cali.run_cali()
    
    

if __name__ == "__main__":
    gui = gui_basics()
    gui.run()

   
   


