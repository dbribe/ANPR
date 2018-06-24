import numpy as np
import os
import math
from skimage import measure
import queue
from ANPR.code.recognize import single_decode
from PIL import Image, ImageEnhance, ImageFile
import matplotlib.pyplot as plt
from ANPR.code.generate_input import regions
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter
from ANPR.code.config import MERGE_STRIDE, REGION_THRESHOLD, LIGHT_THRESHOLD, REGION_STRIDE, FILL_THRESHOLD, IGNORE_THRESHOLD, \
    ROI_LIGHT_THRESHOLD, ROI_AVG_THRESHOLD


ImageFile.LOAD_TRUNCATED_IMAGES = True

total = 0
tot_rects = 0
PLOT = True
SAVEINDEX = 2
SAVE = True
SHOW = False

class Blob(object):
    xa = 1e9
    ya = 1e9
    xb = 0
    yb = 0
    xc1 = 0
    xc2 = 0
    cnt = 0
    yc1 = -1e9
    yc2 = -1e9

def compute_roi(A, img, show):
    global total
    global PLOT
    global tot_rects
    global SAVEINDEX
    global SAVE
    global SHOW

    ans = []

    total += 1

    B = A
    height = np.size(A,0)
    width = np.size(A,1)

    fig, ax = plt.subplots(4)
    plt.gray()


        # ax = [ax]
    # ax[2].imshow(A)
    rect_count = 0

    mr = MERGE_STRIDE

    # print(light_threshold, dark_threshold)
    # A -= np.amin(A)
    # A /= np.amax(A)

    light_threshold = LIGHT_THRESHOLD
    dif = (np.average(B[B>0.5]) - np.average(B[B<0.5])) * (np.amax(B) - np.amin(B))
    if dif > 0.5:
        light_threshold = 0.5

    for i in range(mr-1,height-mr+1):
        for j in range(mr-1,width-mr+1):
            if A[i][j] > light_threshold:
                B[i][j] = min(np.amax(A[i-mr+1:i+mr, j-mr+1:j+mr]) * 1.1, 1)

    ax[0].imshow(img)

    blobs = B > FILL_THRESHOLD * .8
    blobs_labels = measure.label(blobs, background=0, connectivity=1)

    ax[2].imshow(B)
    ax[3].imshow(blobs_labels)
    if PLOT:
        ax[3].imshow(blobs_labels)
    # print(blobs_labels)

    B = blobs_labels
    cnt = np.max(B)
    list = np.empty(cnt+1, dtype=Blob)
    for i in range(cnt+1):
        list[i] = Blob()

    for i in range(np.size(B,0)):
        for j in range(np.size(B,1)):
            index = B[i][j]
            list[index].cnt += 1
            list[index].xa = min(list[index].xa, j)
            list[index].ya = min(list[index].ya, i)
            list[index].xb = max(list[index].xb, j)
            list[index].yb = max(list[index].yb, i)
            if i - j > list[index].yc1 - list[index].xc1:
                list[index].yc1 = i
                list[index].xc1 = j
            if i + j > list[index].yc2 + list[index].xc2:
                list[index].yc2 = i
                list[index].xc2 = j

    for i in range(1, cnt):
        xa = list[i].xa
        ya = list[i].ya
        xb = list[i].xb
        yb = list[i].yb
        xc1 = list[i].xc1
        xc2 = list[i].xc2
        yc1 = list[i].yc1
        yc2 = list[i].yc2
        cnt = list[i].cnt
        rect_width = xb - xa
        rect_height = yb - ya
        if  rect_width >= 60 and rect_height >= 6 and rect_width >= 2 * rect_height \
                and rect_width <= 6 * rect_height and  rect_width >= width / 32 and\
                cnt > rect_width * rect_height / 4:
            nxa = int(xa / width * img.size[0])
            nxb = int(xb / width * img.size[0])
            nya = int(ya / height * img.size[1])
            nyb = int(yb/ height * img.size[1])
            # print(nxa, nya, nxb, nyb, img.size[0], img.size[1])
            cropped = img.rotate(np.arctan((yc2 - yc1) / (xc2 - xc1)) / np.pi * 180, expand=0, center=(nxa + (nxb - nxa)/2, nya + (nyb - nya) / 2)).crop((nxa, nya, nxb, nyb))
            # cropped = img.crop((nxa, nya, nxb, nyb))
            # cropped = cropped.rotate(np.arctan((yc2 - yc1) / (xc2 - xc1)) / np.pi * 180, expand=0)
            smth = int(abs(yc2 - yc1) / (yb - ya) * (nyb - nya) * .3)
            # print("smth is", smth)
            # print(nxa, nya, nxb, nyb)
            cropped = cropped.crop((0, smth, (nxb - nxa), (nyb - nya) - smth))
            cropped = cropped.resize((128, 64), Image.LANCZOS)
            cropped = gaussian_filter(cropped, 0.8)
            cropped = np.asarray(cropped)
            cR = cropped[:, :, 0]
            cG = cropped[:, :, 1]
            cB = cropped[:, :, 2]

            if np.size(cropped, 2) == 4:
                vec = [0.299, 0.587, 0.114, 0]
            else:
                vec = [0.299, 0.587, 0.114]
            c2 = np.dot(cropped, vec)[:, :] / 255

            cropped = 0.7 * np.maximum(np.maximum(cR, cG), cB) + 0.3 * np.minimum(np.minimum(cR, cG), cB)
            cropped = cropped[:, :] / 255
            cropped = cropped - np.amin(cropped)
            cropped = cropped / np.amax(cropped)

            C = cropped[c2 < .5]
            c = cropped[c2 > .5]

            dif = np.average(c) - np.average(C)

            if not math.isnan(dif) and np.average(c) - np.average(C) < .39:
                continue

            c_value = np.amax(cR - (cG + cB) / 2) + \
                np.amax(cG - (cR + cB) / 2) + \
                np.amax(cB - (cR + cG) / 2) + \
                np.amax(cB - cG) + \
                np.amax(cB - cR) + \
                np.amax(cG - cB) + \
                np.amax(cG - cR) + \
                np.amax(cR - cG) + \
                np.amax(cR - cB)

            if c_value < 1800:
                continue

            imsh = cropped
            ax[1].imshow(imsh)
            cropped = np.flip(np.rot90(cropped, -1), 1)
            cropped = np.expand_dims(cropped, axis=2)

            val = single_decode(cropped)
            val = ''.join(val.split('_'))

            rect = Rectangle((nxa, nya), nxb - nxa, nyb - nya, edgecolor='b', fill=False)
            ax[0].add_patch(rect)

            # print(xa / width, ya / height, xb / width, yb / height)

            # print(val)

            if len(val) >= 6:
                ax[1].imshow(imsh)

            if len(val) > 0 and str.isdigit(val[0]):
                val = 'B' + val

            if len(val) < 6:
                continue
            while len(val) > 3 and str.isalpha(val[-4]) and val[-4] != 'O' and val[-4] != 'I':
                val = val[0:len(val) - 1]
            if val[0] == 'F':
                val = 'I' + val
            if (val[0] == 'D' or val[0] == 'E') and str.isdigit(val[1]):
                val = 'B' + val[1:len(val)]
            if str.isalpha(val[0]) and str.isalpha(val[1]) and ((val[0] == 'B' or val[1] == 'B') and not (val[0:2] in regions)):
                val = 'B' + val[2:len(val)]
            if len(val) < 6 or len(val) > 7:
                continue
            if val[0:2] in regions and len(val) == 7:
                for l in range(2,4):
                    # print(val[l])
                    if val[l] == 'I':
                        # print('WUT')
                        val = val[0:l] + '1' + val[l+1:len(val)]
                    if val[l] == 'O':
                        val = val[0:l] + '0' + val[l + 1:len(val)]
                for l in range(4,7):
                    if val[l] == '1':
                        val = val[0:l] + 'I' + val[l+1:len(val)]
                    if val[l] == '0':
                        val = val[0:l] + 'O' + val[l + 1:len(val)]

                if str.isdigit(val[2]) and str.isdigit(val[3]) and str.isalpha(val[4]) and str.isalpha(val[5]) and str.isalpha(val[6]):
                    val = val[0:2] + " " + val[2:4] + " " + val[4:7]
                else:
                    continue
            elif val[0:1] == 'B':
                if len(val) == 6:
                    for l in range(1, 3):
                        if val[l] == 'I':
                            val = val[0:l] + '1' + val[l + 1:len(val)]
                        if val[l] == 'O':
                            val = val[0:l] + '0' + val[l + 1:len(val)]
                    for l in range(3, 6):
                        if val[l] == '1':
                            val = val[0:l] + 'I' + val[l + 1:len(val)]
                        if val[l] == '0':
                            val = val[0:l] + 'O' + val[l + 1:len(val)]

                    if str.isdigit(val[1]) and str.isdigit(val[2]) and str.isalpha(val[3]) and str.isalpha(val[4]) and str.isalpha(val[5]):
                        val = val[0:1] + " " + val[1:3] + " " + val[3:6]
                    else:
                        continue
                else:

                    for l in range(1, 4):
                        if val[l] == 'I':
                            val = val[0:l] + '1' + val[l + 1:len(val)]
                        if val[l] == 'O':
                            val = val[0:l] + '0' + val[l + 1:len(val)]
                    for l in range(4, 7):
                        if val[l] == '1':
                            val = val[0:l] + 'I' + val[l + 1:len(val)]
                        if val[l] == '0':
                            val = val[0:l] + 'O' + val[l + 1:len(val)]
                    if len(val) == 7 and str.isdigit(val[1]) and str.isdigit(val[2]) and str.isdigit(val[3]) and str.isalpha(val[4]) and str.isalpha(val[5]) and str.isalpha(val[6]):
                        val = val[0:1] + " " + val[1:4] + " " + val[4:7]
                    else:
                        continue
            else:
                continue
            # print(val)

            # cropped.save('/Users/Dragonite/Programming/Repos/ANPR/crops/crp' + str(cnt) + '.png')
            if not show:
                ans.append({
                    "boundingBox": [xa / width, ya / height, xb / width, yb / height],
                    "label": val
                })

            if PLOT:
                rect = Rectangle((nxa, nya), nxb - nxa, nyb - nya, edgecolor='springgreen', fill=False)
                ax[0].add_patch(rect)
                ax[0].text(nxa, nya, val,
                        verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(facecolor='black', alpha=0.5),
                        # transform=ax[0].transAxes,
                        color='springgreen', fontsize=9)
            rect_count += 1

    if rect_count:
        if SAVE:
            fig.savefig('/Users/Dragonite/Programming/Repos/translify2/ANPR/detected/' + str(SAVEINDEX) + '/' + str(total) + '.png')
        tot_rects += 1
    else:
        tot_rects += 0
        fig.savefig('/Users/Dragonite/Programming/Repos/translify2/ANPR/undetected/' + str(SAVEINDEX) + '/' + str(
            total) + '.png')
        if SAVE:
            fig.savefig('/Users/Dragonite/Programming/Repos/translify2/ANPR/undetected/' + str(SAVEINDEX) + '/' + str(total) + '.png')
    print(str(tot_rects / total * 100) + '%', tot_rects, total)

    if PLOT and SHOW:
        plt.show()
    plt.close()
    return ans


def show_compute_roi(image_path, show=True):
    img = Image.open(image_path)
    basewidth = 1000
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    contrast = ImageEnhance.Contrast(img)
    mi = np.asarray(img).min()
    ma = np.asarray(img).max()
    contrast_ratio = (305 / (ma - mi + 1))
    contrast_ratio2 = (275 / (ma - mi + 1))

    orig_img = img
    img = contrast.enhance(contrast_ratio)
    contrast2 = ImageEnhance.Contrast(orig_img)
    orig_img = contrast2.enhance(contrast_ratio2)

    img = img.resize((basewidth, hsize), Image.LANCZOS)

    # bw = ImageEnhance.Color(orig_img)
    # orig_img = bw.enhance(0)

    # sharp = ImageEnhance.Sharpness(orig_img)
    # orig_img = sharp.enhance(0.5)

    brightness = ImageEnhance.Brightness(orig_img)
    orig_img = brightness.enhance(1.3)

    o_contrast = ImageEnhance.Contrast(orig_img)
    orig_img = o_contrast.enhance(0.9)

    # sharp = ImageEnhance.Sharpness(orig_img)
    # orig_img = sharp.enhance(1.2)


    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(0.9)
    img = np.asarray(img)

    cR = img[:, :, 0]
    cG = img[:, :, 1]
    cB = img[:, :, 2]

    aR = np.exp(np.average(cR))
    aG = np.exp(np.average(cG))
    aB = np.exp(np.average(cB))

    # print(aR, aG, aB)

    if np.size(img, 2) == 4:
        vec = [0.299*(aG+aB)/(aG+aR+aB)*3/2, 0.587*(aB+aR)/(aG+aR+aB)*3/2, 0.114*(aG+aR)/(aG+aR+aB)*3/2, 0]
        # vec = [0.299, 0.587, 0.114, 0]
        # vec = [.33, .33, .33, 0]
    else:
        vec = [0.299*(aG+aB)/(aG+aR+aB)*3/2, 0.587*(aB+aR)/(aG+aR+aB)*3/2, 0.114*(aG+aR)/(aG+aR+aB)*3/2]
        # vec = [0.299, 0.587, 0.114]
        # vec = [.33, .33, .33]
    img = 1 * np.dot(img, vec) + 0 * np.minimum(np.minimum(img[:,:,0], img[:,:,1]), img[:,:,2])\
        + 0 * np.maximum(np.maximum(img[:,:,0], img[:,:,1]), img[:,:,2])
    final_image = img

    final_image = gaussian_filter(final_image, (np.size(final_image) ** 0.2) / 100)
    final_image = (final_image[:, :] / 255)

    # fig, ax = plt.subplots(1)
    # plt.gray()
    # ax.imshow(img)
    # fig.savefig(
    #     '/Users/Dragonite/Programming/Repos/translify2/ANPR/detected/last.png')

    return compute_roi(final_image, orig_img, show)

# show_compute_roi('huge4.jpg')
# for i in range(1,12):
#     show_compute_roi('big' + str(i) + '.jpg')
# for i in range(2,7):
#     show_compute_roi('images-' + str(i) + '.jpeg')
# for i in range(1,14):
#     show_compute_roi('huge' + str(i) + '.jpg')
# for filename in os.listdir('ANPR/photos/'):
#     show_compute_roi('ANPR/photos/' + filename)
# for filename in os.listdir('undetected/'):
#     show_compute_roi('undetected/' + filename)
# show_compute_roi('pic.png')
# for filename in os.listdir('crops/'):
#     if filename[0] != '.':
#         show_compute_roi('crops/' + filename)
# show_compute_roi('photos/IMG_0385.JPG')

# print(tot_rects)

