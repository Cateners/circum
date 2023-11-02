import cv2
import mediapipe as mp
from PIL import ImageTk, Image
#import pyautogui
import pydirectinput as pyautogui
import tkinter as tk
from tkextrafont import Font
from mediapipe.framework.formats import landmark_pb2
from skspatial.objects import Point, Points, Plane, Vector
import numpy as np
from enum import Enum

class CircumStates(Enum):
    VANILLA = 0
    ADJUST_MOUSE = 1
    MOUSE = 2
    ADJUST_KEYBOARD = 3
    KEYBOARD = 4

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

class Ensure:
    def __init__(self):
        self.isLeftMouseUp = False
        self.isRightMouseUp = False
        self.keyPressed = []
    def leftMouseUp(self):
        if not self.isLeftMouseUp:
            pyautogui.mouseUp()
            self.isLeftMouseUp = True
    def rightMouseUp(self):
        if not self.isRightMouseUp:
            pyautogui.mouseUp(button="right")
            self.isRightMouseUp = True
    def leftMouseDown(self):
        if self.isLeftMouseUp:
            pyautogui.mouseDown()
            self.isLeftMouseUp = False
    def rightMouseDown(self):
        if self.isRightMouseUp:
            pyautogui.mouseDown(button="right")
            self.isRightMouseUp = False
    def leftClick(self):
        self.leftMouseDown()
        self.leftMouseUp()
    def rightClick(self):
        self.rightMouseDown()
        self.rightMouseUp()
    def keyUp(self, key):
        if self.keyPressed.count(key) > 0:
            pyautogui.keyUp(key)
            self.keyPressed.remove(key)
    def keyDown(self, key):
        if self.keyPressed.count(key) == 0:
            pyautogui.keyDown(key)
            self.keyPressed.append(key)




BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def next_frame():

    global current_state, timer, label_text

    # 读取帧
    ret, frame = cap.read()
    image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    #image = np.ones(image.shape, dtype=np.uint8) * 255

    # 处理状态
    match current_state:
        case CircumStates.VANILLA:
            gesture_recognition_result = gesture_recognizer.recognize(mp_image)
            #print(gesture_recognition_result)
            for hand_landmarks in gesture_recognition_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                mp_drawing.draw_landmarks(image, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            label_text.set("确保光线良好，掌心朝向屏幕\n最小化窗口可能导致识别变慢\n✌😀✌鼠标映射、🤟😀🤟键位映射")
            if len(gesture_recognition_result.gestures) > 1:
                if gesture_recognition_result.gestures[0][0].category_name == "Victory" and gesture_recognition_result.gestures[1][0].category_name == "Victory":
                    current_state = CircumStates.ADJUST_MOUSE
                    timer = waiting_time
                elif gesture_recognition_result.gestures[0][0].category_name == "ILoveYou" and gesture_recognition_result.gestures[1][0].category_name == "ILoveYou":
                    current_state = CircumStates.ADJUST_KEYBOARD
                    timer = waiting_time
        case CircumStates.ADJUST_MOUSE:
            global x_min, x_max, y_min, y_max
            temp_x_min = 9999
            temp_x_max = -9999
            temp_y_min = 9999
            temp_y_max = -9999
            gesture_recognition_result = gesture_recognizer.recognize(mp_image)
            for hand_landmarks in gesture_recognition_result.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                mp_drawing.draw_landmarks(image, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                # 获取手部关键点的坐标列表
                points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in hand_landmarks_proto.landmark]
                # 计算最小矩形的左上角和右下角坐标
                temp_x_min = min(min(points, key=lambda p: p[0])[0], temp_x_min)
                temp_y_min = min(min(points, key=lambda p: p[1])[1], temp_y_min)
                temp_x_max = max(max(points, key=lambda p: p[0])[0], temp_x_max)
                temp_y_max = max(max(points, key=lambda p: p[1])[1], temp_y_max)
            # 绘制最小矩形
            cv2.rectangle(image, (temp_x_min, temp_y_min), (temp_x_max, temp_y_max), (0, 0, 255), 1)
            # 绘制区域矩形
            temp_w = temp_x_max - temp_x_min
            temp_h = temp_y_max - temp_y_min
            if temp_w / temp_h >= screen_ratio:
                temp_w = int(screen_ratio * temp_h)
            else:
                temp_h = int(temp_w / screen_ratio)
            x_min = int((temp_x_max + temp_x_min - temp_w) / 2) 
            x_max = x_min + temp_w
            y_min = int((temp_y_max + temp_y_min - temp_h) / 2) 
            y_max = y_min + temp_h
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

            label_text.set(f"双手撑起一个矩形作为控制区域\n倒计时结束后获取鼠标控制权\n{timer//10}")
            timer -= 10

            if timer < 0:
                if x_min == 9999:
                    current_state = CircumStates.VANILLA
                else:
                    current_state = CircumStates.MOUSE
        case CircumStates.MOUSE:
            global last_cursor_x, last_cursor_y, last_gesture
            gesture_recognition_result = gesture_recognizer.recognize(mp_image)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

            current_text = ""

            # 只检测第一只手
            if len(gesture_recognition_result.hand_landmarks) > 0:
                timer = waiting_time

                # 使用第0个点和第5个点的中点作为手部坐标
                img = Image.fromarray(image)
                x_in_image = int((gesture_recognition_result.hand_landmarks[0][0].x + gesture_recognition_result.hand_landmarks[0][5].x) * img.size[0] / 2)
                y_in_image = int((gesture_recognition_result.hand_landmarks[0][0].y + gesture_recognition_result.hand_landmarks[0][5].y) * img.size[1] / 2)
                x_in_control = int((x_in_image - x_min) * screen_width / (x_max - x_min))
                y_in_control = int((y_in_image - y_min) * screen_height / (y_max - y_min))

                # 使用哪只手的坐标呢？
                # 哪只手距离上一次的坐标近就使用哪只手。只可能检测到两只手（已在options中设置）。
                index = 0
                for i, landmarks in enumerate(gesture_recognition_result.hand_landmarks):
                    temp_x_in_image = int((landmarks[0].x + landmarks[5].x) * img.size[0] / 2)
                    temp_y_in_image = int((landmarks[0].y + landmarks[5].y) * img.size[1] / 2)
                    temp_x_in_control = int((temp_x_in_image - x_min) * screen_width / (x_max - x_min))
                    temp_y_in_control = int((temp_y_in_image - y_min) * screen_height / (y_max - y_min))
                    if (temp_x_in_control - last_cursor_x)**2 + (temp_y_in_control - last_cursor_y)**2 < (x_in_control - last_cursor_x)**2 + (y_in_control - last_cursor_y)**2:
                        x_in_image = temp_x_in_image
                        y_in_image = temp_y_in_image
                        x_in_control = temp_x_in_control
                        y_in_control = temp_y_in_control
                        index = i

                cv2.circle(image, (x_in_image, y_in_image), 4, (0, 0, 255), 2)

                if (last_cursor_x - x_in_control)**2 + (last_cursor_y - y_in_control)**2 > step:
                    last_cursor_x = x_in_control
                    last_cursor_y = y_in_control
                
                current_text = f"当前坐标({last_cursor_x},{last_cursor_y})"

                pyautogui.moveTo(last_cursor_x, last_cursor_y)

                gesture = gesture_recognition_result.gestures[index][0].category_name
                match gesture:
                    case "Open_Palm":
                        ensure.leftMouseUp()
                        ensure.rightMouseUp()
                        current_text += "\n抬起鼠标"
                    case "Closed_Fist":
                        ensure.leftMouseDown()
                        current_text += "\n按下左键"
                    case "Victory":
                        ensure.rightMouseDown()
                        current_text += "\n按下右键"
                    case "Pointing_Up":
                        if last_gesture != "Pointing_Up":
                            ensure.leftClick()
                        current_text += "\n左键点击"
                    case "ILoveYou":
                        if last_gesture != "ILoveYou":
                            ensure.rightClick()
                        current_text += "\n右键点击"

                last_gesture = gesture

            else:
                timer -= 10
                current_text = f"没有行为，倒计时{timer//10}后交出鼠标控制权"
                if timer < 0:
                    current_state = CircumStates.VANILLA

            label_text.set(f"只使用一只手哦\n✋放开鼠标、✊按下左键、✌按下右键\n👆左键点击、🤟右键点击\n{current_text}")
        
        case CircumStates.ADJUST_KEYBOARD:
            global standard_plane
            pose_landmarker_result = pose_landmarker.detect(mp_image)
            for pose_landmarks in pose_landmarker_result.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                mp_drawing.draw_landmarks(image, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS, mp_drawing_styles.get_default_pose_landmarks_style())
            label_text.set(f"正在确定你的初始位置\n倒计时结束后获取控制权{timer//10}")
            timer -= 20

            if timer < 0:
                if len(pose_landmarker_result.pose_world_landmarks) > 0:
                    # 记录身体所在平面
                    standard_plane = Plane.best_fit(Points([[pose_landmarker_result.pose_world_landmarks[0][i].x, pose_landmarker_result.pose_world_landmarks[0][i].y, pose_landmarker_result.pose_world_landmarks[0][i].z] for i in [11, 12, 23, 24]]))
                    current_state = CircumStates.KEYBOARD
                else:
                    current_state = CircumStates.VANILLA
        
        case CircumStates.KEYBOARD:
            pose_landmarker_result = pose_landmarker.detect(mp_image)
            for pose_landmarks in pose_landmarker_result.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                mp_drawing.draw_landmarks(image, pose_landmarks_proto, mp_pose.POSE_CONNECTIONS, mp_drawing_styles.get_default_pose_landmarks_style())
            
            def point_of(i):
                return Point([pose_landmarker_result.pose_world_landmarks[0][i].x, pose_landmarker_result.pose_world_landmarks[0][i].y, pose_landmarker_result.pose_world_landmarks[0][i].z])
            
            def diff_z(a, b):
                return abs(pose_landmarker_result.pose_world_landmarks[0][a].z - pose_landmarker_result.pose_world_landmarks[0][b].z)
            

            current_text = ""

            if len(pose_landmarker_result.pose_world_landmarks) > 0:

                timer = waiting_time

                current_plane = Plane.best_fit(Points([[pose_landmarker_result.pose_world_landmarks[0][i].x, pose_landmarker_result.pose_world_landmarks[0][i].y, pose_landmarker_result.pose_world_landmarks[0][i].z] for i in [11, 12, 23, 24]]))
                left_arm_angle = Vector.from_points(point_of(14), point_of(12)).angle_between(
                    Vector.from_points(point_of(14), point_of(16))
                )
                right_arm_angle = Vector.from_points(point_of(13), point_of(11)).angle_between(
                    Vector.from_points(point_of(13), point_of(15))
                )
                # 上键 手向前伸时按下 通过手到肩膀的z距离和手臂夹角判断
                if (diff_z(16, 12) > 0.4 and left_arm_angle > 2.4) or (diff_z(15, 11) > 0.3 and right_arm_angle > 2):
                    ensure.keyDown("up")
                    current_text += "按住上键\n"
                else:
                    ensure.keyUp("up")

                # 下键 手向后缩时按下
                if (diff_z(16, 12) < 0.4 and left_arm_angle < 1.7) or (diff_z(15, 11) < 0.23 and right_arm_angle < 1.4):
                    ensure.keyDown("down")
                    current_text += "按住下键\n"
                else:
                    ensure.keyUp("down")

                #print(current_plane.distance_point(point_of(15)) + abs(pose_landmarker_result.pose_world_landmarks[0][15].y - pose_landmarker_result.pose_world_landmarks[0][11].y)+ abs(pose_landmarker_result.pose_world_landmarks[0][15].x - pose_landmarker_result.pose_world_landmarks[0][11].x))
                #print(left_arm_angle)
                #print(current_plane.distance_point(point_of(16)), current_plane.distance_point(point_of(15)))
                #print(abs(pose_landmarker_result.pose_world_landmarks[0][16].z - pose_landmarker_result.pose_world_landmarks[0][12].z), abs(pose_landmarker_result.pose_world_landmarks[0][15].z - pose_landmarker_result.pose_world_landmarks[0][11].z))
                #print(left_arm_angle, current_plane.distance_point(point_of(16)),abs(pose_landmarker_result.pose_world_landmarks[0][16].z - pose_landmarker_result.pose_world_landmarks[0][12].z), right_arm_angle, current_plane.distance_point(point_of(15)), abs(pose_landmarker_result.pose_world_landmarks[0][15].z - pose_landmarker_result.pose_world_landmarks[0][11].z))
                
                # 左右键，根据身体旋转幅度决定
                if current_plane.vector.angle_between(standard_plane.vector) > 0.5:
                    if pose_landmarker_result.pose_world_landmarks[0][12].z > pose_landmarker_result.pose_world_landmarks[0][11].z:
                        ensure.keyDown("left")
                        ensure.keyUp("right")
                        current_text += "按住左键\n"
                    else:
                        ensure.keyDown("right")
                        ensure.keyUp("left")
                        current_text += "按住右键\n"
                else:
                    ensure.keyUp("left")
                    ensure.keyUp("right")

                # 鼠标左键，在手处于低位时触发
                if pose_landmarker_result.pose_world_landmarks[0][16].y > (4 * pose_landmarker_result.pose_world_landmarks[0][24].y + pose_landmarker_result.pose_world_landmarks[0][12].y)/5:
                    ensure.leftMouseDown()
                    current_text += "按住鼠标左键\n"
                else:
                    ensure.leftMouseUp()

                # 鼠标右键，在手处于低位时触发
                if pose_landmarker_result.pose_world_landmarks[0][15].y > (4 * pose_landmarker_result.pose_world_landmarks[0][23].y + pose_landmarker_result.pose_world_landmarks[0][11].y)/5:
                    ensure.rightMouseDown()
                    current_text += "按住鼠标右键\n"
                else:
                    ensure.rightMouseUp()

            else:
                timer -= 10
                current_text = f"没有行为\n倒计时{timer//10}后交出控制权"
                if timer < 0:
                    current_state = CircumStates.VANILLA

            label_text.set(f"当前操作：{current_text}")




    # 调整图片大小适应窗口
    image = Image.fromarray(image)
    temp_w = image.size[0]
    temp_h = image.size[1]
    ratio = temp_w / temp_h
    w_ratio = root.winfo_width() / root.winfo_height()
    if ratio >= w_ratio:
        temp_w = root.winfo_width()
        temp_h = int(temp_w / ratio)
    else:
        temp_h = root.winfo_height()
        temp_w = int(ratio * temp_h)
    img = ImageTk.PhotoImage(image=image.resize((temp_w, temp_h)))
    label.configure(image=img)
    label.image = img
    root.after(delay, next_frame)

############

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.IMAGE, num_hands=2)

gesture_recognizer = GestureRecognizer.create_from_options(gesture_options)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'))

pose_landmarker = PoseLandmarker.create_from_options(pose_options)


############

# 目前的状态
current_state = CircumStates.VANILLA

# 控制范围
x_min = 9999
y_min = 9999
x_max = -9999
y_max = -9999

# 倒计时
timer = 0
waiting_time = 3000
delay = 10

# 平滑范围
step = 200

ensure = Ensure()

last_gesture = "Victory"

standard_plane = None

############


root = tk.Tk()
root.title("Circum")
root.attributes("-topmost", True)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
screen_ratio = screen_width / screen_height
root.geometry(f"{screen_width//2}x{screen_height//2}+{screen_width//4}+{screen_height//4}")

last_cursor_x = screen_width - 1
last_cursor_y = screen_height - 1

my_font = Font(file="Mengshen-Handwritten.ttf", family="Mengshen-Handwritten")
label_text = tk.StringVar()
label = tk.Label(root, textvariable=label_text, font=("Mengshen-Handwritten", 30), compound="center", anchor="center", image=None, fg="#673AB7")
label.pack(fill="both", expand=True)

root.after(delay, next_frame)

root.mainloop()
############

# 关闭摄像头和窗口
cap.release()
