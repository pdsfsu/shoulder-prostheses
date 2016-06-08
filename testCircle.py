import os
import re
import cv2
import segmentation

# def show_circle(img, img_file_name):
#     temp_img = cv2.medianBlur(img, 5)
#     c_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     circles = segmentation.find_circle(temp_img, 10, 100, 30)
#     ix, iy = 0, 0
#
#     if circles is None:
#         circle_out_str = "_circle_%d-%d.png" % (ix, iy)
#         circle_out_file = re.sub(r'\.jpg', circle_out_str, img_file_name)
#         cv2.imwrite(circle_out_file, c_img)
#         return False
#
#     for i in circles[0, :]:
#         center = (i[0], i[1])
#         ix, iy = i[0], i[1]
#         radius = i[2]
#         circle_color = (0, 255, 0)
#         cv2.circle(c_img, center, radius, circle_color, 1)
#         center_color = (0, 0, 255)
#         cv2.circle(c_img, center, 2, center_color, 1)
#
#     circle_out_str = "_circle_%d-%d.png" % (ix, iy)
#     circle_out_file = re.sub(r'\.jpg', circle_out_str, img_file_name)
#     cv2.imwrite(circle_out_file, c_img)
#     return True


def run_circle_test(img_name_list_file, img_dir):
    names_file = open(img_name_list_file, 'r')
    img_dir = str(img_dir)
    for line in names_file:
        file_name = line.split()
        img_file = img_dir + file_name[0]
        img = cv2.imread(img_file, 0)
        show_circle(img, img_file)
        if img is None:
            print "Image not found: " + img_file
            print os.path.isfile(img_file)
            exit()
    names_file.close()
    return True

