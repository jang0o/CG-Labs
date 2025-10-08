import numpy as np
from PIL import Image,ImageOps

def bresenheim(image, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        derror += dy
        if (derror > (x1-x0)):
            derror -= 2*(x1-x0)
            y += y_update

model_v = []
model_f = []

height, width = 2000,2000

img_matr = np.zeros((height, width, 3), dtype=np.uint8)

file = open('model_1.obj')
for l in file:
    l_split = l.split()

    if l_split[0] == 'v':
        model_v.append([float(l_split[1]), float(l_split[2]), float(l_split[3])])
    if l_split[0] == 'f':
        model_f.append([int(l_split[1].split('/')[0]), int(l_split[2].split('/')[0]), int(l_split[3].split('/')[0])])

for k in range(len(model_f)):
    x0 = int(model_v[model_f[k][0] - 1][0]*9000+1000)
    y0 = int(model_v[model_f[k][0] - 1][1]*9000+1000)
    x1 = int(model_v[model_f[k][1] - 1][0]*9000+1000)
    y1 = int(model_v[model_f[k][1] - 1][1]*9000+1000)
    x2 = int(model_v[model_f[k][2] - 1][0]*9000+1000)
    y2 = int(model_v[model_f[k][2] - 1][1]*9000+1000)
    bresenheim(img_matr, x0, y0, x1, y1, (255, 255, 255))
    bresenheim(img_matr, x0, y0, x2, y2, (255, 255, 255))
    bresenheim(img_matr, x1, y1, x2, y2, (255, 255, 255))

img = Image.fromarray(img_matr, mode="RGB")
img = ImageOps.flip(img)
img.save("img3.png")
img.show()
