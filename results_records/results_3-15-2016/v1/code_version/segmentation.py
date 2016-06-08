import sys
import re
import cv2
import numpy as np

ix,iy = -1,-1

def circle_mask(img, min_dist=20, param_1=100, param_2=30, min_radius=0, max_radius=0):
    global ix,iy
    temp_img = cv2.medianBlur(img, 5)
    zero_img = np.zeros((img.shape), np.uint8)
    dimens = img.shape #returns (y max, x max)
    y_dimen = dimens[0]
    x_dimen = dimens[1]
    max_radius = (int)(.25 * x_dimen)
    param_2 = 30
    dp = 1
    circles = cv2.HoughCircles(temp_img, cv2.cv.CV_HOUGH_GRADIENT, 1, min_dist, param1=100, param2=param_2, minRadius=0,maxRadius=max_radius)

    while circles is None and param_2 > 5:
        param_2 -= 5
        circles = cv2.HoughCircles(temp_img, cv2.cv.CV_HOUGH_GRADIENT, 1, min_dist, param1=100, param2=param_2, minRadius=0, maxRadius=max_radius)

    if circles is None:
        return zero_img

    while len(circles[0]) > 1:
        max_radius = (int)(.9 * max_radius)
        circles = cv2.HoughCircles(temp_img, cv2.cv.CV_HOUGH_GRADIENT, 1, min_dist, param1=100, param2=param_2, minRadius=0, maxRadius=max_radius)

    if circles is None:
        return zero_img

    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        center = (i[0],i[1])
        ix,iy = i[0],i[1]
        print ix,iy
        radius = i[2]
        fill_color = 255
        cv2.circle(zero_img, center, (int) ((1.25) * radius), fill_color, thickness=-1)

    return zero_img


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

def addEdge(img):
    canny_img = cv2.Canny(img,100,200)
    edge_intensity = 255
    ret, mask = cv2.threshold(canny_img, 10, edge_intensity, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    combined_img = cv2.bitwise_and(img, img, mask = mask_inv)
    return combined_img

def unsmoothed(img, eqImg, seed, th, img_file, out_info):
    img_n = img
    img_h = eqImg

    out_img_n = seeded_region_growing(img_n, seed, th)
    out_str_n = out_info + '_n.png'
    out_file_n = re.sub(r'\.jpg', out_str_n, img_file)
    cv2.imwrite(out_file_n, out_img_n)

    out_img_h = seeded_region_growing(img_h, seed, th)
    out_str_h = out_info + '_h.png'
    out_file_h = re.sub(r'\.jpg', out_str_h, img_file)
    cv2.imwrite(out_file_h, out_img_h)


def averageFilters(img, eqImg, seed, th, img_file, out_info):
    img_a = cv2.blur(img, (3,3))
    img_ah = cv2.blur(eqImg, (3,3))

    out_img_a = seeded_region_growing(img_a, seed, th)
    out_str_a = out_info + '_a.png'
    out_file_a = re.sub(r'\.jpg', out_str_a, img_file)
    cv2.imwrite(out_file_a, out_img_a)

    out_img_ah = seeded_region_growing(img_ah, seed, th)
    out_str_ah = out_info + '_ah.png'
    out_file_ah = re.sub(r'\.jpg', out_str_ah, img_file)
    cv2.imwrite(out_file_ah, out_img_ah)


def GaussianFilters(img, eqImg, seed, th, img_file, out_info):
    img_g = cv2.GaussianBlur(img, (3,3), 0)
    img_gh = cv2.GaussianBlur(eqImg, (3,3), 0)

    out_img_g = seeded_region_growing(img_g, seed, th)
    out_str_g = out_info + '_g.png'
    out_file_g = re.sub(r'\.jpg', out_str_g, img_file)
    cv2.imwrite(out_file_g, out_img_g)

    out_img_gh = seeded_region_growing(img_gh, seed, th)
    out_str_gh = out_info + '_gh.png'
    out_file_gh = re.sub(r'\.jpg', out_str_gh, img_file)
    cv2.imwrite(out_file_gh, out_img_gh)


def BilateralFilters(img, eqImg, seed, th, img_file, out_info):
    img_b = cv2.bilateralFilter(img, 5, 75, 75)
    img_bh = cv2.bilateralFilter(eqImg, 5, 75, 75)

    out_img_b = seeded_region_growing(img_b, seed, th)
    out_str_b = out_info + '_b.png'
    out_file_b = re.sub(r'\.jpg', out_str_b, img_file)
    cv2.imwrite(out_file_b, out_img_b)

    out_img_bh = seeded_region_growing(img_bh, seed, th)
    out_str_bh = out_info + '_bh.png'
    out_file_bh = re.sub(r'\.jpg', out_str_bh, img_file)
    cv2.imwrite(out_file_bh, out_img_bh)

# http://stackoverflow.com/questions/5775352/python-return-2-ints-for-index-in-2d-lists-given-item
def index2D(arr, val):
    for i, x in enumerate(arr):
        if val in x:
            return (i, x.index(val))

def multithresholdOtsu(img, mask=None):

    lval = 256
    # compute normalized histogram
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    cumHist = np.cumsum(hist)
    print "cumHist[255]: %f" % cumHist[255]

    norm_hist = hist.ravel()/cumHist[255]

    cumSum = np.cumsum(np.array(norm_hist), dtype=float)
    m = [norm_hist[i] * i for i in range(len(norm_hist))]
    means = np.array(m)
    cumMean = np.cumsum(means, dtype=float)
    mg =cumMean[lval - 1]
    global_variance = 0
    for i in range(len(norm_hist)):
        global_variance += ((i - mg)**2 * norm_hist[i])

    between_class_variance = np.zeros((256, 256))
    max_variance = -1
    print "cumsum[255]: %f" % cumSum[255]

    for k1 in range(1, lval - 3):
        p1 = cumSum[k1]
        for k2 in range(k1 + 1, lval - 2):
            p2 = cumSum[k2] - p1
            p3 = 1 - p1 - p2

            m1 = (1/p1) * cumMean[k1] if p1 > 0 else 0
            m2 = (1/p2) * (cumMean[k2] - cumMean[k1]) if p2 > 0 else 0
            m3 = (1/p3) * (cumMean[lval - 1] - cumMean[k2]) if p3 > 0 else 0
            if between_class_variance[k1][k2] == 0:
                between_class_variance[k1][k2] = (p1 * (m1 - mg)**2) + (p2 * (m2 - mg)**2) + (p3 * (m3 - mg)**2)
                if between_class_variance[k1][k2] > max_variance:
                    max_variance = between_class_variance[k1][k2]

    th1 = 0
    th2 = 0
    maxs = 0

    # find max variances and sum for averages if more than 1 for each
    for i in range(1, lval - 3):
        for j in range(i + 1, lval -2):
            if between_class_variance[i][j] == max_variance:
                th1 += i
                th2 += j
                maxs += 1

    th1 = th1 / maxs
    th2 = th2 / maxs
    print "Number of maxs: %d" % (maxs)
    print "Max variance: %d" % (max_variance)
    return (th1, th2)

def threeThresholdOtsu(img, mask=None):

    lval = 256
    # compute normalized histogram
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    cumHist = np.cumsum(hist)
    print "cumHist[255]: %f" % cumHist[255]

    norm_hist = hist.ravel()/cumHist[255]

    cumSum = np.cumsum(np.array(norm_hist), dtype=float)
    m = [norm_hist[i] * i for i in range(len(norm_hist))]
    means = np.array(m)
    cumMean = np.cumsum(means, dtype=float)
    mg =cumMean[lval - 1]
    global_variance = 0
    for i in range(len(norm_hist)):
        global_variance += ((i - mg)**2 * norm_hist[i])

    between_class_variance = np.zeros((256, 256, 256))
    max_variance = -1
    print "cumsum[255]: %f" % cumSum[255]

    for k1 in range(1, lval - 4):
        p1 = cumSum[k1]
        for k2 in range(k1 + 1, lval - 3):
            p2 = cumSum[k2] - cumSum[k1]
            for k3 in range(k2 + 1, lval - 2):
                p3 = cumSum[k3] - cumSum[k2]
                p4 = 1 - p1 - p2 - p3

                m1 = (1/p1) * cumMean[k1] if p1 > 0 else 0
                m2 = (1/p2) * (cumMean[k2] - cumMean[k1]) if p2 > 0 else 0
                m3 = (1/p3) * (cumMean[k3] - cumMean[k2]) if p3 > 0 else 0
                m4 = (1/p4) * (cumMean[lval - 1] - cumMean[k3]) if p4 > 0 else 0
                if between_class_variance[k1][k2][k3] == 0:
                    between_class_variance[k1][k2][k3] = (p1 * (m1 - mg)**2) + (p2 * (m2 - mg)**2) + (p3 * (m3 - mg)**2) + (p4 * (m4 -mg)**2)
                    if between_class_variance[k1][k2][k3] > max_variance:
                        max_variance = between_class_variance[k1][k2][k3]

    th1 = 0
    th2 = 0
    th3 = 0
    maxs = 0

    # find max variances and sum for averages if more than 1 for each
    for i in range(1, lval - 4):
        for j in range(i + 1, lval - 3):
            for k in range(j + 1, lval - 2):
                if between_class_variance[i][j][k] == max_variance:
                    th1 += i
                    th2 += j
                    th3 += k
                    maxs += 1

    th1 = th1 / maxs
    th2 = th2 / maxs
    th3 = th3 / maxs
    print "Number of maxs: %d" % (maxs)
    print "Max variance: %d" % (max_variance)
    return (th1, th2, th3)


# Otsu with 2 thresholds, extend to multi if needed later
def otsu(img, img_file, mask=None):

    blur = cv2.GaussianBlur(img, (5,5),0)

    thresholds = multithresholdOtsu(blur,mask)
    th1 = thresholds[0]
    th2 = thresholds[1]


    if mask is None:
        ret, thresh1 = cv2.threshold(blur,th1,255,cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(blur,th2,255,cv2.THRESH_BINARY_INV)
    else:
        combined_img = cv2.bitwise_and(blur, blur, mask=mask)
        ret, thresh1 = cv2.threshold(combined_img,th1,255,cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(combined_img,th2,255,cv2.THRESH_BINARY_INV)

    out_img_o = cv2.bitwise_and(thresh1, thresh2, mask=None)
    out_info_o = "_otsu_%d-%d" % (th1, th2)
    out_str_o = out_info_o + '.png'
    out_file_o = re.sub(r'\.jpg', out_str_o, img_file)
    cv2.imwrite(out_file_o, out_img_o)

def otsuThree(img, img_file, mask=None):

    blur = cv2.GaussianBlur(img, (5,5),0)

    thresholds = threeThresholdOtsu(blur,mask)
    th1 = thresholds[0]
    th2 = thresholds[1]
    th3 = thresholds[2]


    if mask is None:
        ret, thresh1 = cv2.threshold(blur,th1,150,cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(blur,th2,150,cv2.THRESH_BINARY_INV)
        lower_img = cv2.bitwise_and(thresh1, thresh2, mask=None)
        ret, thresh2 = cv2.threshold(blur, th2, 200, cv2.THRESH_BINARY)
        ret, thresh3 = cv2.threshold(blur, th3, 200, cv2.THRESH_BINARY_INV)
        middle_img = cv2.bitwise_and(thresh2, thresh3, mask=None)
        ret, uppper_img = cv2.threshold(blur, th3, 255, cv2.THRESH_BINARY)
    else:
        combined_img = cv2.bitwise_and(blur, blur, mask=mask)
        ret, thresh1 = cv2.threshold(combined_img,th1,150,cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(combined_img,th2,150,cv2.THRESH_BINARY_INV)
        lower_img = cv2.bitwise_and(thresh1, thresh2, mask=mask)
        ret, thresh2 = cv2.threshold(blur, th2, 200, cv2.THRESH_BINARY)
        ret, thresh3 = cv2.threshold(blur, th3, 200, cv2.THRESH_BINARY_INV)
        middle_img = cv2.bitwise_and(thresh2, thresh3, mask=mask)
        ret, uppper_img = cv2.threshold(blur, th3, 255, cv2.THRESH_BINARY)


    out_img_one = cv2.bitwise_or(lower_img, middle_img, mask=mask)
    out_img_o = cv2.bitwise_or(out_img_one, uppper_img, mask=mask)

    out_info_o = "_otsu_%d-%d-%d" % (th1, th2, th3)
    out_str_o = out_info_o + '.png'
    out_file_o = re.sub(r'\.jpg', out_str_o, img_file)
    cv2.imwrite(out_file_o, out_img_o)

# Begin main
arg_list = sys.argv

if len(arg_list) < 2:
    sys.exit("Missing input parameters")


img_file = arg_list[1]

if len(arg_list) > 2:
    th = int(arg_list[2])
else:
    th = 5

img = cv2.imread(img_file, 0)
c_mask = circle_mask(img)
combined_img = cv2.bitwise_and(img, img, mask=circle_mask(img))
seed = (ix, iy)

h = cv2.equalizeHist(combined_img)
dimens = img.shape #returns (y max, x max)
height = dimens[0]
width = dimens[1]
out_info = "_results_%d_%d-%d" % (th, ix, iy)
mask_out_str = "_circle_mask_%d-%d.png" % (ix, iy)
mask_out_file = re.sub(r'\.jpg', mask_out_str, img_file)

cv2.imwrite(mask_out_file, combined_img)
unsmoothed(combined_img, h, seed, th, img_file, out_info)
averageFilters(combined_img, h, seed, th, img_file, out_info)
GaussianFilters(combined_img, h, seed, th, img_file, out_info)
BilateralFilters(combined_img, h, seed, th, img_file, out_info)
otsu(img, img_file, c_mask)
otsuThree(img, img_file, c_mask)

print "done"
