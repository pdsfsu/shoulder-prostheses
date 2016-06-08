import os
import sys
import re
import cv2
import numpy as np
import evaluation
import morphologicalOps

ix,iy = -1,-1

def get_seed():
    global ix,iy
    x,y = ix,iy
    return x,y

def find_circle(img, min_dist=10, param_1=100, param_2=30):
    dimens = img.shape #returns (y max, x max)
    x_dimen = dimens[1]
    max_radius = (int)(.5 * x_dimen)
    min_radius = (int)(.08 * x_dimen)
    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, min_dist, param1=param_1, param2=param_2,
                               minRadius=min_radius,maxRadius=max_radius)
    if circles is None and param_2 > 5:
        min_radius = (int)(.05 * x_dimen)
        while circles is None and param_2 > 5:
            param_2 -= 5
            circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, min_dist, param1=param_1, param2=param_2,
                                       minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return circles

    while len(circles[0]) > 1:
        c_len = len(circles[0])
        max_radius = (int)(.9 * max_radius)
        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, min_dist, param1=param_1, param2=param_2,
                                   minRadius=min_radius, maxRadius=max_radius)
        if circles is None:
            return circles
        elif len(circles[0]) == c_len:
            min_radius = (int)(1.05 * min_radius)

    circles = np.uint16(np.around(circles))
    return circles

def circle_mask(img, min_dist=10, param_1=100, param_2=30):
    global ix,iy
    temp_img = cv2.medianBlur(img, 5)
    zero_img = np.zeros((img.shape), np.uint8)
    circles = find_circle(temp_img, 10, 100, 30)

    if circles is None:
        return zero_img

    for i in circles[0,:]:
        center = (i[0],i[1])
        ix,iy = i[0],i[1]
        # print ix,iy
        radius = i[2]
        fill_color = 255
        cv2.circle(zero_img, center, (int) ((1.25) * radius), fill_color, thickness=-1)

    return zero_img

def show_circle(img, img_file_name):
    temp_img = cv2.medianBlur(img, 5)
    c_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = find_circle(temp_img, 10, 100, 30)
    ix, iy = 0, 0

    if circles is None:
        circle_out_str = "_circle_%d-%d.png" % (ix, iy)
        circle_out_file = re.sub(r'\.jpg', circle_out_str, img_file_name)
        cv2.imwrite(circle_out_file, c_img)
        return False

    for i in circles[0, :]:
        center = (i[0], i[1])
        ix, iy = i[0], i[1]
        radius = i[2]
        circle_color = (0, 255, 0)
        cv2.circle(c_img, center, radius, circle_color, 1)
        center_color = (0, 0, 255)
        cv2.circle(c_img, center, 2, center_color, 1)

    circle_out_str = "_circle_%d-%d.png" % (ix, iy)
    circle_out_file = re.sub(r'\.jpg', circle_out_str, img_file_name)
    cv2.imwrite(circle_out_file, c_img)
    return True

def seeded_region_growing(img, seed, threshold=5):
    seg_img = np.zeros((img.shape), np.uint8)
    dimens = img.shape #returns (y max, x max)
    y_dimen = dimens[0]
    x_dimen = dimens[1]
    size = 1
    pix_area = y_dimen*x_dimen

    contour = [] # will be [ [[x1, y1],..., [[xn, yn]] ]
    max_difference = threshold
    neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
    cur_pix = [seed[0], seed[1]]
    seg_img[cur_pix[1], cur_pix[0]] = 255

    while(size <= pix_area):
        for j in range(4):
            temp_pix = [cur_pix[0] + neighbors[j][0], cur_pix[1] + neighbors[j][1]]

            #check if it belongs to the image
            is_in_img = x_dimen>temp_pix[0]>0 and y_dimen>temp_pix[1]>0
            #taken if not already selected before
            if (is_in_img and (seg_img[temp_pix[1], temp_pix[0]]==0)):
                difference = abs(int(img[cur_pix[1], cur_pix[0]]) - int(img[temp_pix[1], temp_pix[0]]))
                if (difference < max_difference):
                    contour.append(temp_pix)
                    seg_img[temp_pix[1], temp_pix[0]] = 150

        size += 1
        seg_img[cur_pix[1], cur_pix[0]] = 255
        if (len(contour) > 0):
            cur_pix = contour.pop()
        else:
            size = pix_area + 1

    return seg_img

def unsmoothed(img, seed, th, img_file, out_info, man_img):
    img_n = img
    # SRG result image
    out_img_n = seeded_region_growing(img_n, seed, th)
    # Morphological op
    kernel = morphologicalOps.open_elliptical_kernel()
    out_img_n = morphologicalOps.opening(out_img_n, kernel)

    out_str_n = out_info + '_n.png'
    out_file_n = re.sub(r'\.jpg', out_str_n, img_file)
    cv2.imwrite(out_file_n, out_img_n)
    t = evaluation.findTotals(out_img_n, man_img)
    f = open('n_all.txt', 'a')
    f.write(img_file + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " " + str(t[3]) + "\n")
    f.close()

def bilateral_filter(img, seed, th, img_file, out_info, man_img):
    img_b = cv2.bilateralFilter(img, 5, 5, 5)
    #  SRG result image
    out_img_b = seeded_region_growing(img_b, seed, th)
    # morphological ops
    kernel = morphologicalOps.open_elliptical_kernel()
    out_img_b = morphologicalOps.opening(out_img_b, kernel)

    out_str_b = out_info + '_b.png'
    out_file_b = re.sub(r'\.jpg', out_str_b, img_file)
    cv2.imwrite(out_file_b, out_img_b)
    t = evaluation.findTotals(out_img_b, man_img)
    f = open('b_all.txt', 'a')
    f.write(img_file + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " " + str(t[3]) + "\n")
    f.close()

def single_threshold_otsu(img, mask=None):
    lval = 256
    # compute normalized histogram
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    cumHist = np.cumsum(hist)
    norm_hist = hist.ravel()/cumHist[255]
    cumSum = np.cumsum(np.array(norm_hist), dtype=float)
    m = [norm_hist[i] * i for i in range(len(norm_hist))]
    means = np.array(m)
    cumMean = np.cumsum(means, dtype=float)
    mg =cumMean[lval - 1]
    global_variance = 0
    for i in range(len(norm_hist)):
        global_variance += ((i - mg)**2 * norm_hist[i])
    between_class_variance = np.zeros(256)
    max_variance = -1
    for k in range(1, lval - 2):
        p1 = cumSum[k]
        p2 = 1 - p1
        m1 = (1/p1) * cumMean[k] if p1 > 0 else 0
        m2 = (1/p2) * (cumMean[lval - 1] - cumMean[k]) if p2 > 0 else 0
        if between_class_variance[k] == 0:
            between_class_variance[k] = (p1 * (m1 - mg)**2) + (p2 * (m2 - mg)**2)
            if between_class_variance[k] > max_variance:
                    max_variance = between_class_variance[k]
    th = 0
    maxs = 0
    # find max variances and sum for averages if more than 1 for each
    for i in range(1, lval - 1):
        if between_class_variance[i] == max_variance:
            th += i
            maxs += 1
    th = th / maxs
    return th

def otsu_one(img, img_file, man_img, mask=None):
    # cv2.GaussianBlur(img, (5,5),0)
    # Bilateral Filter is used separately so it can be adjusted for each type of segmentation
    blur = cv2.bilateralFilter(img, 5, 5, 5)
    th = single_threshold_otsu(blur, mask)
    if mask is None:
        ret, thresh = cv2.threshold(blur,th,255,cv2.THRESH_BINARY)
    else:
        combined_img = cv2.bitwise_and(blur, blur, mask=mask)
        ret, thresh = cv2.threshold(combined_img,th,255,cv2.THRESH_BINARY)
    # Otsu result image
    out_img_o = thresh
    # morphological ops
    kernel = morphologicalOps.open_elliptical_kernel()
    out_img_o = morphologicalOps.opening(out_img_o, kernel)

    out_info_o = "_otsu_%d" % (th)
    out_str_o = out_info_o + '.png'
    out_file_o = re.sub(r'\.jpg', out_str_o, img_file)
    cv2.imwrite(out_file_o, out_img_o)
    t = evaluation.findTotals(out_img_o, man_img)
    f = open('o1_all.txt', 'a')
    f.write(img_file + " " + str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + " " + str(t[3]) + "\n")
    f.close()


# Begin main
arg_list = sys.argv

if len(arg_list) < 2:
    sys.exit("Missing input parameters")

img_dir = arg_list[1]
name_file = 'image_file_names.txt'
circle_only = False
th = 5
if len(arg_list) > 2:
    name_file = arg_list[2]
    if len(arg_list) > 3:
        circle_only = arg_list[3]
        if len(arg_list) > 4:
            th = int(arg_list[4])

nfile = open(name_file, 'r')
for line in nfile:
    file_name = line.split()
    img_file = img_dir + file_name[0]

    img = cv2.imread(img_file, 0)

    if img is None:
        print "Image not found: " + img_file
        print os.path.isfile(img_file)
        exit()

    show_circle(img, img_file)

    if circle_only == False:
        man_str = "_manual.png"
        man_file = re.sub(r'\.jpg', man_str, img_file)
        man_img = cv2.imread(man_file, 0)

        c_mask = circle_mask(img)
        combined_img = cv2.bitwise_and(img, img, mask=c_mask)
        masked_man = cv2.bitwise_and(man_img, man_img, mask=c_mask)
        seed = (ix, iy)

        dimens = img.shape #returns (y max, x max)
        height = dimens[0]
        width = dimens[1]
        out_info = "_results_%d_%d-%d" % (th, ix, iy)
        mask_out_str = "_circle_mask_%d-%d.png" % (ix, iy)
        mask_out_file = re.sub(r'\.jpg', mask_out_str, img_file)
        cv2.imwrite(mask_out_file, combined_img)

        masked_man_str = "_masked_manual.png"
        masked_man_file = re.sub(r'\.jpg', masked_man_str, img_file)
        cv2.imwrite(masked_man_file, masked_man)
        unsmoothed(combined_img, seed, th, img_file, out_info, masked_man)
        bilateral_filter(combined_img, seed, th, img_file, out_info, masked_man)
        otsu_one(img, img_file, masked_man, c_mask)

nfile.close()
print "done"
