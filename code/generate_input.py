import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import THREE_DIGITS_THRESHOLD
from random import choice
from string import ascii_uppercase

regions = ['B', 'IS', 'PH', 'CJ', 'CT', 'TM', 'DJ', 'SV', 'BC', 'AG', 'BH', 'MS', 'GL', 'BV', 'DB', 'NT', 'MM', 'BZ', 'OT', 'AR', 'HD', 'BT', 'VS', 'SB', 'VL', 'TR', 'IF', 'GJ', 'AB', 'VN', 'SM', 'BR', 'HR', 'BN', 'CS', 'CL', 'GR', 'IL', 'MH', 'SJ', 'TL', 'CV']
populations = [1883425, 772348, 762886, 691106, 684082, 683540, 660544, 634810, 616168, 612431, 575398, 540508, 530612, 529906, 510287, 507399, 472117, 440347, 421769, 412235, 410383, 404429, 393340, 377273, 374240, 369897, 353481, 345771, 342336, 339510, 336117, 316652, 304765, 301425, 287535, 287269, 267147, 265559, 259212, 225631, 211622, 211254]

def build_data(count, type='', show_only=True):
    total = sum(populations)
    current = 0
    last = 0
    for i in range(len(populations)):
        current += populations[i]
        crt_count = int(count * current / total) - last
        last += crt_count
        region = regions[i]
        for j in range(crt_count):
            name = ''.join(choice(ascii_uppercase) for i in range(3))
            if regions[i] == 'B' and random.random() < THREE_DIGITS_THRESHOLD:
                number = random.randint(1, 999)
                if number < 100 and number > 9:
                    number = '0' + str(number)
                elif number < 10:
                    number = '00' + str(number)
            else:
                number = random.randint(1,99)
                if number < 10:
                    number = '0' + str(number)
            string = region + ' ' + str(number) + ' ' + name
            img = Image.new('RGB', (315, 63), color=(250, 250, 250))
            fnt = ImageFont.truetype('car_license_font.ttf', int(62 + random.random() * 4))
            d = ImageDraw.Draw(img)
            d.text((6, -6), string, fill=(0, 0, 0), font=fnt)
            img = img.resize((128, 64), Image.LANCZOS)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1 + random.random() * 0.8)
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(2)
            img = img.rotate(random.random() * 9 - 4.5, expand=0)
            a = 1
            b = 0
            c = -random.random() * 5.5  # left/right (i.e. 5/-5)
            d = 0
            e = 1
            f = -random.random() * 3
            img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
            img = img.filter(ImageFilter.GaussianBlur(0.9 + random.random() * 0.6))
            brightness = ImageEnhance.Brightness(img)
            # img = img.resize((128, 64), Image.LANCZOS)
            img = brightness.enhance(1 + random.random() * 0.3)
            img = img.crop((0, random.random() * 2, 128 - random.random() * 3, 64 + (random.random() * 2.5 - 0.5) * 6))
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1 + (random.random() * 3 - 1) * 0.2)
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(1 + (random.random() * 2 - 1) * 0.4)
            img = img.filter(ImageFilter.GaussianBlur(0.1 + random.random() * 0.65))
            if show_only:
                img.show()
            else:
                img.save('dataset/plate/' + type + region + str(number) + name + '.png')

# build_data(10, 'test/')

build_data(100, 'test7/', False)
build_data(5000, 'train7/', False)
