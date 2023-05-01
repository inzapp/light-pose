import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


g_angle_range = 45
g_augment_count = 4
g_generated_dir = './generated'

g_angle_table = []
g_angle_table_init = False


def convert_angle(angle):
    global g_angle_table
    if not g_angle_table_init:
        for i in range(4):
            for j in range(90):
                if j <= 45:
                    g_angle_table.append(j)
                else:
                    g_angle_table.append(45 - (j - 45))
        g_angle_table.append(0)
    return g_angle_table[abs(angle)]


def main():
    global g_generated_dir
    os.makedirs(g_generated_dir, exist_ok=True)
    image_paths = glob('**/*.jpg', recursive=True)

    angles = list(np.linspace(-g_angle_range, g_angle_range, g_augment_count, dtype=np.int32))
    if 0 in angles:
        angles.remove(0)
        print(f'found and remove zero angle, new augment_count : {len(angles)}')

    for image_path in tqdm(image_paths):
        basename = os.path.basename(image_path)[:-4]
        raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_h, img_w = raw.shape[:2]
        origin_h, origin_w = img_h, img_w
        label_path = f'{image_path[:-4]}.txt'

        size = img_h if img_h >= img_w else img_w
        raw = cv2.resize(raw, (size, size), interpolation=cv2.INTER_LINEAR)
        img_h, img_w = raw.shape[:2]

        center_point = (raw.shape[1] // 2, raw.shape[0] // 2)
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        for i, angle in enumerate(angles):
            scale = np.cos(np.deg2rad(convert_angle(angle)))
            m = np.asarray(cv2.getRotationMatrix2D(center_point, angle, scale))
            img = cv2.warpAffine(raw, m, (0, 0))
            img = cv2.resize(img, (origin_w, origin_h), interpolation=cv2.INTER_AREA)

            label = ''
            for line in lines:
                confidence, x_pos, y_pos = list(map(float, line.split()))
                x_pos *= img_w
                y_pos *= img_h
                new_x = m[0][0] * x_pos + m[0][1] * y_pos + m[0][2]
                new_y = m[1][0] * x_pos + m[1][1] * y_pos + m[1][2]
                new_x /= img_w
                new_y /= img_h
                new_x, new_y = np.clip([new_x, new_y], 0.0, 1.0)
                label += f'{confidence:.6f} {new_x:.6f} {new_y:.6f}\n'
            cv2.imwrite(f'{g_generated_dir}/{basename}_{i}.jpg', img)
            with open(f'{g_generated_dir}/{basename}_{i}.txt', 'wt') as f:
                f.writelines(label)


if __name__ == '__main__':
    main()

