import numpy as np
from PIL import Image, ImageOps
import random

H, W = 2000, 2000
img_arr = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf, dtype=float)

file = open('model_1.obj')

model = []

for line in file:
    elements = line.split()

    if elements[0] == 'v':
        arr = []
        for coord in elements[1:]:
            arr.append(float(coord))
        model.append(arr)

file.seek(0)

model_v = []
model_f = []

for i in file:
    i_split = i.split()

    if i_split[0] == 'v':
        model_v.append([float(i_split[1]), float(i_split[2]), float(i_split[3])])
    if i_split[0] == 'f':
        model_f.append([int(i_split[1].split('/')[0]), int(i_split[2].split('/')[0]), int(i_split[3].split('/')[0])])

def baric_coords(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return (lambda0, lambda1, lambda2)

def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, color, z_buffer):
    xmin = min(int(x0), int(x1), int(x2))
    if xmin < 0: xmin = 0
    xmax = max(int(x0), int(x1), int(x2))+1

    ymin = min(int(y0), int(y1), int(y2))
    if ymin < 0: ymin = 0
    ymax = max(int(y0), int(y1), int(y2))+1

    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            coord = baric_coords(x, y, x0, y0, x1, y1, x2, y2)
            lambd0, lambd1, lambd2 = coord

            if lambd0 >= 0 and lambd1 >= 0 and lambd2 >= 0:
                z_val = lambd0 * z0 + lambd1 * z1 + lambd2 * z2

                if z_val < z_buffer[y, x]:
                    image[y, x] = color
                    z_buffer[y, x] = z_val

# draw_triangle( 200, 200, 400, 400, 600, 400, img_arr)
# img = Image.fromarray(img_arr)
# img.save("img1.png")
# img.show()

def normal(v0, v1, v2):
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)

    v1_v0 = v1 - v0
    v2_v0 = v2 - v0

    n = np.cross(v1_v0, v2_v0)
    n /= np.linalg.norm(n)
    return n

def draw_polygon(model_v, model_f, image):
    light_dir = np.array([0, 0, 1])

    for i in model_f:

        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)

        color = (red, green, blue)

        x0, y0, z0 = 10000 * model_v[i[0]-1][0] + 1000, 10000 * model_v[i[0]-1][1] + 1000, 10000 * model_v[i[0]-1][2] + 1000
        v0 = (x0, y0, z0)
        x1, y1, z1 = 10000 * model_v[i[1] - 1][0] + 1000, 10000 * model_v[i[1] - 1][1] + 1000, 10000 * model_v[i[1]-1][2] + 1000
        v1 = (x1, y1, z1)
        x2, y2, z2 = 10000 * model_v[i[2] - 1][0] + 1000, 10000 * model_v[i[2] - 1][1] + 1000, 10000 * model_v[i[2]-1][2] + 1000
        v2 = (x2, y2, z2)


        x = normal(v0, v1, v2)
        if (x[2] < 0):
            color = (-255 * x[2], 0, 0)
            draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, color, z_buffer)
        else:
            continue

draw_polygon(model_v, model_f, img_arr)


img = Image.fromarray(img_arr)
image = ImageOps.flip(img)
image.save("img3.png")
image.show()
