import numpy as np
import os
import queue
from recognize import single_decode
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter
from config import MERGE_STRIDE, REGION_THRESHOLD, LIGHT_THRESHOLD, REGION_STRIDE, FILL_THRESHOLD, IGNORE_THRESHOLD, \
    ROI_LIGHT_THRESHOLD, ROI_AVG_THRESHOLD

cnt = 0
tot_rects = 0
PLOT = True

def main():


    def compute_roi(A, img):
        global cnt
        global PLOT
        global tot_rects
        cnt += 1
        # A = gaussian_filter(A, 2)
        visited = np.zeros(np.shape(A))
        B = A
        height = np.size(A,0)
        width = np.size(A,1)
        max_queue_size = height * width

        if PLOT:
            fig, ax = plt.subplots(3)
            plt.gray()
        # ax = [ax]
        rect_count = 0
        # ax.imshow(A)

        # ax[1].imshow(B)
        mr = MERGE_STRIDE

        # print(light_threshold, dark_threshold)

        light_threshold = LIGHT_THRESHOLD

        for i in range(2,height-2):
            for j in range(2,width-2):
                if A[i][j] > light_threshold:
                    B[i][j] = min(np.amax(A[i-mr+1:i+mr, j-mr+1:j+mr]) * 1.3, 1)

        if PLOT:
            ax[0].imshow(img)
            ax[2].imshow(B)
        st = REGION_STRIDE

        for ii in range((int)(height / st)):
            for jj in range((int)(width / st)):
                i = ii * st
                j = jj * st
                if not visited[i][j] and B[i][j] > FILL_THRESHOLD:
                    ya = i
                    xa = j
                    yb = i
                    xb = j
                    mi = B[i][j]
                    ma = B[i][j]
                    qx = queue.Queue(max_queue_size)
                    qy = queue.Queue(max_queue_size)
                    qx.put(j)
                    qy.put(i)
                    visited[i][j] = 1
                    count = 1
                    ignored_count = 0
                    total = 0
                    xc1 = j
                    yc1 = i
                    xc2 = j
                    yc2 = i
                    while not qx.empty() and not qy.empty():
                        # print('eliminated')
                        xc = qx.get()
                        yc = qy.get()
                        for [dx, dy] in [[0, st], [st, 0], [0, -st], [-st, 0]]:
                            # print(dx)
                            # print(dy)
                            xn = xc + dx
                            yn = yc + dy
                            # print(yn,xn)
                            if xn < 0 or xn >= width or yn < 0 or yn >= height or \
                                    B[yn][xn] - mi > REGION_THRESHOLD or ma - B[yn][xn] > REGION_THRESHOLD or \
                                    visited[yn][xn] == 1:
                                continue
                            total += B[yn][xn]
                            visited[yn][xn] = 1
                            qx.put(xn)
                            qy.put(yn)
                            count += 1
                            # print('added')
                            xa = min(xa, xn)
                            ya = min(ya, yn)
                            xb = max(xb, xn)
                            yb = max(yb, yn)
                            mi = min(mi, B[yn][xn])
                            ma = max(ma, B[yn][xn])
                            if yn - xn > yc1 - xc1:
                                yc1 = yn
                                xc1 = xn
                            if yn + xn > yc2 + xc2:
                                yc2 = yn
                                xc2 = xn
                    # print('adding rect')
                    # print(xb-xa, yb-ya)
                    rect_width = xb - xa
                    rect_height = yb - ya
                    # rect = Rectangle((xa, ya), xb - xa, yb - ya, edgecolor='b', fill=False)
                    # ax[0].add_patch(rect)
                    # rect_c

                    if total / (count - ignored_count) >= ROI_AVG_THRESHOLD and ma > ROI_LIGHT_THRESHOLD \
                            and rect_width >= 40 and rect_height >= 5 and rect_width >= 2.5 * rect_height \
                            and rect_width <= 7 * rect_height and \
                            count > rect_width * rect_height / (st * st * 4) and rect_width >= width / 32:
                        nxa = int(xa / width * img.size[0])
                        nxb = int(xb / width * img.size[0])
                        nya = int(ya / height * img.size[1])
                        nyb = int(yb/ height * img.size[1])
                        # print(nxa, nya, nxb, nyb, img.size[0], img.size[1])
                        cropped = img.crop((nxa, nya, nxb, nyb))
                        cropped = cropped.rotate(np.arctan((yc2 - yc1) / (xc2 - xc1)) / np.pi * 180, expand=0)
                        smth = int(abs(yc2 - yc1) / (yb - ya) * (nyb - nya) * .3)
                        print("smth is", smth)
                        print(nxa, nya, nxb, nyb)
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
                        if len(val) < 1:
                            continue
                        begins_alpha = str.isalpha(val[0])
                        ends_alpha = str.isalpha(val[-1])
                        has_digit = 0
                        for a in val:
                            if str.isdigit(a):
                                has_digit += 1
                        # cropped.save('/Users/Dragonite/Programming/Repos/ANPR/crops/crp' + str(cnt) + '.png')
                        if begins_alpha and ends_alpha and has_digit >= 2 and has_digit <= 3 and len(val) >= 6 and len(val) <= 7:
                            if PLOT:
                                rect = Rectangle((nxa, nya), nxb - nxa, nyb - nya, edgecolor='springgreen', fill=False)
                                ax[0].add_patch(rect)
                                ax[0].text(nxa, nya, val,
                                        verticalalignment='bottom', horizontalalignment='left',
                                        # transform=ax[0].transAxes,
                                        color='springgreen', fontsize=9)
                                ax[1].imshow(imsh)
                            rect_count += 1
        # print(rect_count)
        tot_rects += rect_count
        print(tot_rects, cnt)
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
        contrast_ratio = (285 / (ma - mi + 1))

        orig_img = img = contrast.enhance(contrast_ratio)

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
        img = np.array(img)
        img = np.dot(img, [0.299, 0.587, 0.114])
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
    # show_compute_roi('pic.png')
    # for filename in os.listdir('crops/'):
    #     if filename[0] != '.':
    #         show_compute_roi('crops/' + filename)

    print(tot_rects)

if __name__ == "__main__":
    main()
