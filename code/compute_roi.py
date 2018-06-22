import numpy as np
import os
from skimage import measure
import queue
from recognize import single_decode
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from generate_input import regions
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter
from config import MERGE_STRIDE, REGION_THRESHOLD, LIGHT_THRESHOLD, REGION_STRIDE, FILL_THRESHOLD, IGNORE_THRESHOLD, \
    ROI_LIGHT_THRESHOLD, ROI_AVG_THRESHOLD

total = 0
tot_rects = 0
PLOT = True
SAVEINDEX = 2
SAVE = False

class Blob(object):
    xa = 1e9
    ya = 1e9
    xb = 0
    yb = 0
    xc1 = 0
    xc2 = 0
    yc1 = -1e9
    yc2 = -1e9

def main():


    def compute_roi(A, img):
        global total
        global PLOT
        global tot_rects
        global SAVEINDEX
        global SAVE
        total += 1
        # A = gaussian_filter(A, 2)
        visited = np.zeros(np.shape(A))
        B = A
        height = np.size(A,0)
        width = np.size(A,1)
        max_queue_size = height * width

        fig, ax = plt.subplots(1)
        if PLOT:

            plt.gray()
            ax = [ax]
        rect_count = 0
        # ax.imshow(A)

        mr = MERGE_STRIDE

        # print(light_threshold, dark_threshold)

        light_threshold = LIGHT_THRESHOLD
        for i in range(2,height-2):
            for j in range(2,width-2):
                if A[i][j] > light_threshold:
                    B[i][j] = min(np.amax(A[i-mr+1:i+mr, j-mr+1:j+mr]) * 1.3, 1)

        ax[0].imshow(img)

        blobs = B > FILL_THRESHOLD
        blobs_labels = measure.label(blobs, background=0, connectivity=1)

        # ax[2].imshow(B)
        # ax[3].imshow(blobs_labels)
        # if PLOT:
        #     ax[3].imshow(blobs_labels)
        # print(blobs_labels)

        B = blobs_labels
        cnt = np.max(B)
        list = np.empty(cnt+1, dtype=Blob)
        for i in range(cnt+1):
            list[i] = Blob()

        for i in range(np.size(B,0)):
            for j in range(np.size(B,1)):
                index = B[i][j]
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
            rect_width = xb - xa
            rect_height = yb - ya
            if  rect_width >= 30 and rect_height >= 3 and rect_width >= 1.5 * rect_height \
                    and rect_width <= 7 * rect_height and  rect_width >= width / 32:
                nxa = int(xa / width * img.size[0])
                nxb = int(xb / width * img.size[0])
                nya = int(ya / height * img.size[1])
                nyb = int(yb/ height * img.size[1])
                # print(nxa, nya, nxb, nyb, img.size[0], img.size[1])
                cropped = img.crop((nxa, nya, nxb, nyb))
                cropped = cropped.rotate(np.arctan((yc2 - yc1) / (xc2 - xc1)) / np.pi * 180, expand=0)
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
                cropped = np.maximum(np.maximum(cR, cG), cB)
                cropped = cropped[:, :] / 255
                imsh = cropped
                cropped = np.flip(np.rot90(cropped, -1), 1)
                cropped = np.expand_dims(cropped, axis=2)
                val = single_decode(cropped)
                print(val)
                val = ''.join(val.split('_'))

                rect = Rectangle((nxa, nya), nxb - nxa, nyb - nya, edgecolor='b', fill=False)
                ax[0].add_patch(rect)
                if len(val) >= 6:
                    ax[1].imshow(imsh)

                if len(val) > 0 and str.isdigit(val[0]):
                    val = 'B' + val
                if len(val) < 6:
                    continue
                if not str.isalpha(val[-1]) or not str.isalpha(val[-2]) or not str.isalpha(val[-3]):
                    continue
                while len(val) > 3 and str.isalpha(val[-4]):
                    val = val[0:len(val) - 1]
                if val[0] == 'F':
                    val = 'I' + val
                if (val[0] == 'D' or val[0] == 'E') and str.isdigit(val[1]):
                    val = 'B' + val[1:len(val)]
                if str.isalpha(val[0]) and str.isalpha(val[1]) and (val[0] == 'B' or val[1] == 'B' and not (val[0:2] in regions)):
                    val = 'B' + val[2:len(val)]
                if len(val) < 6 or len(val) > 7:
                    continue
                if val[0:2] in regions and len(val) == 7:
                    if str.isdigit(val[2]) and str.isdigit(val[3]) and str.isalpha(val[4]) and str.isalpha(val[5]) and str.isalpha(val[6]):
                        val = val
                    else:
                        continue
                elif val[0:1] == 'B':
                    if len(val) == 6:
                        if str.isdigit(val[1]) and str.isdigit(val[2]) and str.isalpha(val[3]) and str.isalpha(val[4]) and str.isalpha(val[5]):
                            val = val
                        else:
                            continue
                    elif len(val) == 7 and str.isdigit(val[1]) and str.isdigit(val[2]) and str.isdigit(val[3]) and str.isalpha(val[4]) and str.isalpha(val[5]) and str.isalpha(val[6]):
                        val = val
                    else:
                        continue
                else:
                    continue


                # cropped.save('/Users/Dragonite/Programming/Repos/ANPR/crops/crp' + str(cnt) + '.png')
                if PLOT:
                    rect = Rectangle((nxa, nya), nxb - nxa, nyb - nya, edgecolor='springgreen', fill=False)
                    ax[0].add_patch(rect)
                    ax[0].text(nxa, nya, val,
                            verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(facecolor='black', alpha=0.5),
                            # transform=ax[0].transAxes,
                            color='springgreen', fontsize=9)
                    # ax[1].imshow(imsh)
                rect_count += 1

        if rect_count:
            if SAVE:
                fig.savefig('/Users/Dragonite/Programming/Repos/ANPR/detected/' + str(SAVEINDEX) + '/' + str(total) + '.png')
            tot_rects += 1
        else:
            tot_rects += 0
            if SAVE:
                img.save('/Users/Dragonite/Programming/Repos/ANPR/undetected/' + str(SAVEINDEX) + '/' + str(total) + '.png')
        print(str(tot_rects / total * 100) + '%', tot_rects, total)

        if PLOT:
            plt.show()
        plt.close()


    def show_compute_roi(image_path):
        img = Image.open(image_path)
        basewidth = 500
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

        if img.size[0] > basewidth:
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
        # img = np.array(img)
        img1 = np.dot(img, [0.299, 0.587, 0.114])
        img = np.asarray(img)
        cR = img[:, :, 0]
        cG = img[:, :, 1]
        cB = img[:, :, 2]
        img2 = np.minimum(np.minimum(cR, cG), cB)
        img = 1 * img1 + 0 * img2
        final_image = img
        final_image = gaussian_filter(final_image, (np.size(final_image) ** 0.2) / 100)
        final_image = (final_image[:, :] / 255)

        compute_roi(final_image, orig_img)

    # show_compute_roi('huge4.jpg')
    # for i in range(1,12):
    #     show_compute_roi('big' + str(i) + '.jpg')
    # for i in range(2,7):
    #     show_compute_roi('images-' + str(i) + '.jpeg')
    # for i in range(1,14):
    #     show_compute_roi('huge' + str(i) + '.jpg')
    for filename in os.listdir('photos/'):
        show_compute_roi('photos/' + filename)
    # for filename in os.listdir('undetected/'):
    #     show_compute_roi('undetected/' + filename)
    # show_compute_roi('pic.png')
    # for filename in os.listdir('crops/'):
    #     if filename[0] != '.':
    #         show_compute_roi('crops/' + filename)

    print(tot_rects)

if __name__ == "__main__":
    main()
