import cv2
import os
from os.path import splitext

masks_dir = r'F:/datasets/brainMRI/masks/'
images_dir = r'F:/datasets/brainMRI/imgs/'
output_dir = r'F:/datasets/brainMRI/output/'

for cur_dir, cur_dir_subdir, cur_dir_files in os.walk(images_dir):
    for i in range(len(cur_dir_files)):  # 这个len()出来会比真正存在的图片数多一个, 因为有.DS_Store文件

        # 若cur_dir_files[i]指代文件a.txt, 那么splitext(cur_dir_files[i])会返回一个tuple, 即('a', '.txt')
        # if splitext(cur_dir_files[i])[0] == ".DS_Store":
        #     print("第" + str(i) + "个文件为.DS_Store, 不执行打印")
        #     continue

        file_name = splitext(cur_dir_files[i])[0] + "." + "tif"  # 确保全部改成png格式后缀

        image = cv2.imread(images_dir + file_name)
        mask = cv2.imread(masks_dir + file_name)

        binary_mask = cv2.Canny(mask, 30, 100)  # Canny边缘检测算法, 不懂
        mask_contour = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 获取轮廓, 不懂
        cv2.drawContours(image, mask_contour[0], -1, (139, 69, 19), 1)  # 在原图上画masks轮廓, 不懂

        cv2.imwrite(output_dir + file_name, image)
        # print("已打印第" + str(i) + "张图片.")


if __name__== '__main__':
    image = cv2.imread(r'C:\Users\yangjie\Desktop\this\firstwork\imgs\img.png')
    mask = cv2.imread(r'C:\Users\yangjie\Desktop\this\firstwork\imgs\mask.png')

    # file_name = splitext(cur_dir_files[i])[0] + "." + "tif"

    binary_mask = cv2.Canny(mask, 30, 100)  # Canny边缘检测算法, 不懂
    mask_contour = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 获取轮廓, 不懂
    # cv2.drawContours(image, mask_contour[0], -1, (0, 0, 255), 1)  # 在原图上画masks轮廓, 不懂
    cv2.drawContours(image, mask_contour[0], -1, (0, 255, 0), 1)
    cv2.imwrite(r'C:\Users\yangjie\Desktop\this\firstwork\imgs' + r'\final.png', image)
