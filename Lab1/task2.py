from PIL import Image
import numpy as np
from math import cos,sin,pi,sqrt

img_matr = np.zeros((200,200,3), dtype=np.uint8)
def draw_line(img, x0, y0, x1, y1, color):
    count = 100
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img[y, x] = color

def draw_line2(img, x0, y0, x1, y1, color):
    count = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img[y, x] = color

def draw_line3(image, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        image[y, x] = color

def draw_line4(image, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        image[y, x] = color

def draw_line5(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        # image[y, x] = color
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line6(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        # image[y, x] = color
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color

def draw_line7(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1-x0)):
            derror -= 2.0*(x1-x0)
            y += y_update



for k in range(13):
    x0,y0 = 100,100
    x1 = int(100 + 95 * cos((2*pi/13)*k))
    y1 = int(100 + 95 * sin((2*pi/13)*k))
    # draw_line(img_matr, x0, y0, x1, y1, (255,255,255))
    # draw_line2(img_matr, x0, y0, x1, y1, (255,255,255 ))
    # draw_line3(img_matr, x0, y0, x1, y1, (255,255,255))
    # draw_line4(img_matr, x0, y0, x1, y1, (255, 255, 255))
    # draw_line5(img_matr, x0, y0, x1, y1, (255, 255, 255))
    # draw_line6(img_matr, x0, y0, x1, y1, (255, 255, 255))
    draw_line7(img_matr, x0, y0, x1, y1, (255, 255, 255))



img = Image.fromarray(img_matr, mode="RGB")
img.save("img.png")
img.show()

