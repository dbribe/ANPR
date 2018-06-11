import numpy as np
import os
import queue
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter
from config import MERGE_STRIDE, REGION_THRESHOLD, LIGHT_THRESHOLD, REGION_STRIDE, FILL_THRESHOLD, IGNORE_THRESHOLD, \
    ROI_LIGHT_THRESHOLD, ROI_AVG_THRESHOLD

cnt = 0
def main():


    def compute_roi(A, img):
        global cnt
        # A = gaussian_filter(A, 2)
        visited = np.zeros(np.shape(A))
        B = A
        height = np.size(A,0)
        width = np.size(A,1)
        max_queue_size = height * width


        fig, ax = plt.subplots(1)
        plt.gray()
        ax = [ax]
        rect_count = 0
        # ax.imshow(A)

        # ax[1].imshow(B)
        mr = MERGE_STRIDE

        light_threshold_position = (int) (0.6 * np.size(A, 0) * np.size(A, 1))
        dark_threshold_position = (int) (0.5 * np.size(A, 0) * np.size(A, 1))
        light_threshold = np.partition(np.reshape(A, np.size(A, 0) * np.size(A, 1)), light_threshold_position)[light_threshold_position]
        dark_threshold = np.partition(np.reshape(A, np.size(A, 0) * np.size(A, 1)), dark_threshold_position)[dark_threshold_position]

        print(light_threshold, dark_threshold)

        light_threshold -= 0.05
        dark_threshold += 0.05

        light_threshold = LIGHT_THRESHOLD

        for i in range(height):
            for j in range(width):
                if A[i][j] > light_threshold:
                    for k in range(i, i+mr):
                        for l in range(j, j+mr):
                            if k >= 0 and k < height and l >= 0 and l < width:
                                B[i][j] = min(max(B[i][j], A[k][l] * 1.3), 1)
        ax[0].imshow(B)
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
                        nyb = int(yb / height * img.size[1])
                        print(nxa, nya, nxb, nyb, img.size[0], img.size[1])
                        cropped = img.crop((nxa, nya, nxb, nyb))
                        cropped = cropped.resize((128, 64), Image.LANCZOS)
                        cnt += 1
                        cropped.save('/Users/Dragonite/Programming/Repos/ANPR/crops/crp' + str(cnt) + '.png')
                        rect = Rectangle((xa, ya), xb - xa, yb - ya, edgecolor='b', fill=False)
                        ax[0].add_patch(rect)
                        rect_count += 1
        print(rect_count)
        plt.show()

    def show_compute_roi(image_path):
        img = Image.open(image_path)
        basewidth = 500
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        if img.size[0] > basewidth:
            img = img.resize((basewidth, hsize), Image.LANCZOS)
        contrast = ImageEnhance.Contrast(img)
        mi = np.asarray(img).min()
        ma = np.asarray(img).max()
        contrast_ratio = (325 / (ma - mi + 1))

        orig_img = img = contrast.enhance(contrast_ratio)

        bw = ImageEnhance.Color(orig_img)
        orig_img = bw.enhance(0)

        sharp = ImageEnhance.Sharpness(orig_img)
        orig_img = sharp.enhance(0.5)

        brightness = ImageEnhance.Brightness(orig_img)
        orig_img = brightness.enhance(2)

        o_contrast = ImageEnhance.Contrast(orig_img)
        orig_img = o_contrast.enhance(2)

        sharp = ImageEnhance.Sharpness(orig_img)
        orig_img = sharp.enhance(2)

        img = np.asarray(img)
        img = np.dot(img, [0.299, 0.587, 0.114])
        final_image = np.array(img)
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


if __name__ == "__main__":
    main()
