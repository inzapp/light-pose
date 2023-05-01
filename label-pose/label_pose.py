import os
import cv2
import numpy as np

from enum import Enum, auto


g_win_size = (416, 768)
g_win_name = 'LabelPose v2.1 by Inzapp'


class Limb(Enum):
    HEAD = 0
    NECK = auto()
    RIGHT_SHOULDER = auto()
    RIGHT_ELBOW = auto()
    RIGHT_WRIST = auto()
    LEFT_SHOULDER = auto()
    LEFT_ELBOW = auto()
    LEFT_WRIST = auto()
    RIGHT_HIP = auto()
    RIGHT_KNEE = auto()
    RIGHT_ANKLE = auto()
    LEFT_HIP = auto()
    LEFT_KNEE = auto()
    LEFT_ANKLE = auto()


class LabelPose:
    def __init__(self):
        self.image_paths = self.init_image_paths()
        if len(self.image_paths) == 0:
            print('No image files in path.')
            exit(0)
        self.raw = None
        self.guide_img = None
        self.show_skeleton = True
        self.cur_image_path = ''
        self.cur_label_path = ''
        self.max_limb_size = len(Limb)
        self.limb_index = 0
        self.cur_label = self.reset_label()
        self.guide_label = self.reset_label()
        self.font_scale = 0.5
        self.text_positions = self.init_text_positions()
        

    def init_image_paths(self):
        import natsort
        import tkinter as tk
        from glob import glob
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askdirectory()
        print(image_path)
        image_paths = natsort.natsorted(glob(f'{image_path}/*.jpg'))
        for i in range(len(image_paths)):
            image_paths[i] = image_paths[i].replace('\\', '/')
        return image_paths

    def init_text_positions(self):
        text_positions = []
        for i, limb in enumerate(list(Limb)):
            tx = 0
            ty = 10 + (i * 15 + 2)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(text=limb.name, fontFace=font_face, fontScale=self.font_scale, thickness=1)
            tw, th = text_size
            tx1 = tx
            ty1 = ty - th
            tx2 = tx + tw
            ty2 = ty + 1
            text_positions.append([tx, ty, tx1, ty1, tx2, ty2])  # put_text_tx, put_text_ty, rect_tx1, rect_ty1, rect_tx2, rect_ty2
        return text_positions

    def resize(self, img, size):
        img_height, img_width = img.shape[:2]
        if img_width > size[0] or img_height > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def reset_label(self):
        return [[0, 0, 0] for _ in range(self.max_limb_size)]  # [confidence, x_pos, y_pos]

    def circle(self, img, x, y, emphasis=False):
        img = cv2.circle(img, (x, y), 8, (128, 255, 128), thickness=-1, lineType=cv2.LINE_AA)
        img = cv2.circle(img, (x, y), 4, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
        if emphasis:
            img = cv2.circle(img, (x, y), 16, (128, 255, 128), thickness=-1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (x, y), 8, (32, 32, 192), thickness=-1, lineType=cv2.LINE_AA)
        return img

    def line_if_valid(self, img, p1, p2):
        if p1[0] == 1 and p2[0] == 1:
            img = cv2.line(img, (p1[1], p1[2]), (p2[1], p2[2]), (64, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        return img

    def get_limb_guide_img(self, cur_x, cur_y):
        global g_win_size
        img = self.guide_img.copy()
        img = self.circle(img, self.guide_label[self.limb_index][1], self.guide_label[self.limb_index][2])
        thickness = 1
        for i, limb in enumerate(list(Limb)):
            if i == self.limb_index:
                font_face = cv2.FONT_HERSHEY_DUPLEX
                color = (255, 255, 255)
            else:
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                color = (128, 128, 128)
            tx, ty, tx1, ty1, tx2, ty2 = self.text_positions[i]
            if i != self.limb_index and tx1 + g_win_size[0] <= cur_x <= tx2 + g_win_size[0] and ty1 <= cur_y <= ty2:
                color = (224, 224, 224)
            img = cv2.putText(img, limb.name, (tx, ty), fontFace=font_face, fontScale=self.font_scale, color=color, lineType=cv2.LINE_AA, thickness=thickness)
        return img

    def update(self, cur_x=-1, cur_y=-1):
        global g_win_name
        img = self.raw.copy()
        if self.show_skeleton:
            img = self.line_if_valid(img, self.cur_label[Limb.HEAD.value], self.cur_label[Limb.NECK.value])

            img = self.line_if_valid(img, self.cur_label[Limb.NECK.value], self.cur_label[Limb.RIGHT_SHOULDER.value])
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_SHOULDER.value], self.cur_label[Limb.RIGHT_ELBOW.value])
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_ELBOW.value], self.cur_label[Limb.RIGHT_WRIST.value])

            img = self.line_if_valid(img, self.cur_label[Limb.NECK.value], self.cur_label[Limb.LEFT_SHOULDER.value])
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_SHOULDER.value], self.cur_label[Limb.LEFT_ELBOW.value])
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_ELBOW.value], self.cur_label[Limb.LEFT_WRIST.value])

            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_HIP.value], self.cur_label[Limb.LEFT_HIP.value])

            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_SHOULDER.value], self.cur_label[Limb.RIGHT_HIP.value])
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_HIP.value], self.cur_label[Limb.RIGHT_KNEE.value])
            img = self.line_if_valid(img, self.cur_label[Limb.RIGHT_KNEE.value], self.cur_label[Limb.RIGHT_ANKLE.value])

            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_SHOULDER.value], self.cur_label[Limb.LEFT_HIP.value])
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_HIP.value], self.cur_label[Limb.LEFT_KNEE.value])
            img = self.line_if_valid(img, self.cur_label[Limb.LEFT_KNEE.value], self.cur_label[Limb.LEFT_ANKLE.value])
        limb_index = self.get_text_index_if_cursor_in_text(cur_x, cur_y)
        for i, label in enumerate(self.cur_label):
            use, x, y = label
            if use == 1:
                img = self.circle(img, x, y, i == limb_index)
        img = np.append(img, self.get_limb_guide_img(cur_x, cur_y), axis=1)
        cv2.imshow(g_win_name, img)

    def save_label(self):
        global g_win_size
        label_content = ''
        for use, x, y in self.cur_label:
            x = x / float(g_win_size[0] - 1)
            y = y / float(g_win_size[1] - 1)
            label_content += f'{use:.1f} {x:.6f} {y:.6f}\n'
        with open(self.cur_label_path, 'wt') as f:
            f.writelines(label_content)

    def load_label_if_exists(self, guide=False):
        global g_win_size
        label_path = './guide.txt' if guide else self.cur_label_path
        if os.path.exists(label_path) and os.path.isfile(label_path):
            with open(label_path, 'rt') as f:
                lines = f.readlines()
            for i in range(self.max_limb_size):
                use, x, y = list(map(float, lines[i].split()))
                if guide:
                    self.guide_label[i] = [int(use), int(x * float(g_win_size[0])), int(y * float(g_win_size[1]))]
                else:
                    self.cur_label[i] = [int(use), int(x * float(g_win_size[0])), int(y * float(g_win_size[1]))]
        if not guide:
            self.update()

    def is_cursor_in_image(self, x, y):
        global g_win_size
        return g_win_size[0] > x and g_win_size[1] > y

    def find_not_labeled_image_index(self):
        print(f'start to find not labeled file')
        for i in range(len(self.image_paths)):
            label_path = f'{self.image_paths[i][:-4]}.txt'
            if os.path.exists(label_path) and os.path.isfile(label_path):
                with open(label_path, 'rt') as f:
                    lines = f.readlines()
                not_labeled = True
                for line in lines:
                    confidence, x, y = list(map(float, line.split()))
                    if confidence == 1.0:
                        not_labeled = False
                        break
                if not_labeled:
                    print(f'not labeled file found. index : {i}')
                    return i
        print(f'not labeled file not found')
        return -1

    def run(self):
        global g_win_name, g_win_size
        index = 0
        cv2.namedWindow(g_win_name)
        cv2.setMouseCallback(g_win_name, self.mouse_callback)
        self.guide_img = self.resize(cv2.imdecode(np.fromfile('./guide.jpg', dtype=np.uint8), cv2.IMREAD_COLOR), g_win_size)
        self.load_label_if_exists(guide=True)
        print_log = True
        while True:
            self.cur_image_path = self.image_paths[index]
            if print_log:
                print(f'[{index}] : {self.cur_image_path}')
            print_log = True
            self.cur_label_path = f'{self.cur_image_path[:-4]}.txt'
            self.raw = self.resize(cv2.imdecode(np.fromfile(self.cur_image_path, dtype=np.uint8), cv2.IMREAD_COLOR), g_win_size)
            self.load_label_if_exists()
            self.update()
            while True:
                res = cv2.waitKey(0)
                if res == ord('d'):  # go to next if input key was 'd'
                    self.save_label()
                    if index == len(self.image_paths) - 1:
                        print('Current image is last image')
                    else:
                        self.limb_index = 0
                        self.cur_label = self.reset_label()
                        index += 1
                        break
                elif res == ord('a'):  # go to previous image if input key was 'a'
                    self.save_label()
                    if index == 0:
                        print('Current image is first image')
                    else:
                        self.limb_index = 0
                        self.cur_label = self.reset_label()
                        index -= 1
                        break
                elif res == ord('w'):  # toggle show skeleton
                    self.show_skeleton = not self.show_skeleton
                    break
                elif res == ord('e'):  # go to next limb
                    self.limb_index += 1
                    if self.limb_index == self.max_limb_size:
                        self.limb_index = 0
                    print_log = False
                    print(f'limb index : {self.limb_index}')
                    break
                elif res == ord('q'):  # go to prev limb
                    self.limb_index -= 1
                    if self.limb_index == -1:
                        self.limb_index = self.max_limb_size - 1
                    print_log = False
                    print(f'limb index : {self.limb_index}')
                    break
                elif res == ord('f'):  # auto find not labeled image
                    not_labeled_index = self.find_not_labeled_image_index()
                    if not_labeled_index != -1:
                        self.cur_label = self.reset_label()
                        index = not_labeled_index
                    break
                elif res == ord('x'):  # remove cur label
                    self.cur_label = self.reset_label()
                    self.save_label()
                    break
                elif res == 27:  # exit if input key was ESC
                    self.save_label()
                    exit(0)

    def get_text_index_if_cursor_in_text(self, x, y):
        global g_win_size
        for i, position in enumerate(self.text_positions):
            _, _, tx1, ty1, tx2, ty2 = position
            if tx1 + g_win_size[0] <= x <= tx2 + g_win_size[0] and ty1 <= y <= ty2:
                return i
        return -1

    def mouse_callback(self, event, x, y, flag, _):
        if event == 0 and flag == 0:  # no click mouse moving
            self.update(x, y)
        elif event == 4 and flag == 0:  # left click end
            if self.is_cursor_in_image(x, y):
                self.cur_label[self.limb_index] = [1, x, y]
                self.limb_index += 1
                if self.limb_index == self.max_limb_size:
                    self.limb_index = 0
                self.update()
                self.save_label()
            else:
                limb_index = self.get_text_index_if_cursor_in_text(x, y)
                if limb_index != -1:
                    self.limb_index = limb_index
                    self.update()
                    self.save_label()
        elif event == 5 and flag == 0:  # right click end
            if not self.is_cursor_in_image(x, y):
                return
            self.cur_label[self.limb_index] = [0, 0, 0]
            self.update()
            self.save_label()


if __name__ == '__main__':
    LabelPose().run()
