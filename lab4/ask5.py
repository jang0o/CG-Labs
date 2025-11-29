import numpy as np
from PIL import Image, ImageOps
import random
from math import *

H, W = 2000, 2000
u0, v00 = W / 2, H / 2
a_x, a_y = 2500, 2500
img_arr = np.zeros((H, W, 3), dtype=np.uint8)
z_buffer = np.full((H, W), np.inf, dtype=float)

file = open('model_1.obj')

model = []
texture_coords = [] # vt
faces = [] # номера

for line in file:
    elements = line.split()

    if elements[0] == 'v':
        arr = []
        for coord in elements[1:]:
            arr.append(float(coord))
        model.append(arr)
    elif elements[0] == 'vt':
        arr = []
        for coord in elements[1:]:
            arr.append(float(coord))
        texture_coords.append(arr)

file.seek(0)

model_v = []
model_f = []
model_ft = []
model_vt = []

for i in file:
    i_split = i.split()

    if i_split[0] == 'v':
        model_v.append([float(i_split[1]), float(i_split[2]), float(i_split[3])])
    if i_split[0] == 'f':
        model_f.append([int(i_split[1].split('/')[0]), int(i_split[2].split('/')[0]), int(i_split[3].split('/')[0])])
        model_ft.append([int(i_split[1].split('/')[1]), int(i_split[2].split('/')[1]), int(i_split[3].split('/')[1])])
    if i_split[0] == 'vt':
        model_vt.append([float(i_split[1]), float(i_split[2])])

tex_img = Image.open("bunny-atlas.jpg")
tex_arr = np.array(tex_img)

H_t = tex_arr.shape[0]
W_t = tex_arr.shape[1]

def baric_coords(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return (lambda0, lambda1, lambda2)


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, image, vt0, vt1, vt2, I0, I1, I2, z_buffer):
    xmin = min(int(x0), int(x1), int(x2))
    if xmin < 0: xmin = 0
    xmax = max(int(x0), int(x1), int(x2)) + 1

    ymin = min(int(y0), int(y1), int(y2))
    if ymin < 0: ymin = 0
    ymax = max(int(y0), int(y1), int(y2)) + 1

    color = np.array([255,100,255])



    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            coord = baric_coords(x, y, x0, y0, x1, y1, x2, y2)
            lambd0, lambd1, lambd2 = coord

            if lambd0 >= 0 and lambd1 >= 0 and lambd2 >= 0:
                z_val = lambd0 * z0 + lambd1 * z1 + lambd2 * z2

                if z_val < z_buffer[y, x]:
                    u_tex = lambd0 * vt0[0] + lambd1 * vt1[0] + lambd2 * vt2[0]
                    v_tex = lambd0 * vt0[1] + lambd1 * vt1[1] + lambd2 * vt2[1]

                    tex1 = int(W_t * u_tex)
                    tex2 = int(H_t * (1-v_tex))

                    color = tex_arr[tex2,tex1]
                    # I = -(lambd0 * I0 + lambd1 * I1 + lambd2 * I2)
                    # if I < 0:
                    #     I = 0
                    image[y, x] = color # (color*I).astype(np.uint8)
                    z_buffer[y, x] = z_val


def normal(v0, v1, v2):
    v0 = np.array(v0)
    v1 = np.array(v1)
    v2 = np.array(v2)

    v1_v0 = v1 - v0
    v2_v0 = v2 - v0

    n = np.cross(v1_v0, v2_v0)
    n /= np.linalg.norm(n)
    return n


def turn(x, y, z, a, b, g, tx=0.0, ty=0.0, tz=0.0):
    R = np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]]).dot(
        [[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]).dot(
        np.array([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]]))

    p = np.array([x, y, z])
    preob = R.dot(p) + np.array([tx, ty, tz])
    return preob

for i in range(len(model_v)):
    model_v[i][0], model_v[i][1], model_v[i][2] = turn(model_v[i][0], model_v[i][1], model_v[i][2], 0, pi / 2, 0,
                                                       ty=-0.03, tz=0.5)
ver_norm = [np.array([0.0, 0.0, 0.0]) for t in range(len(model_v))]
ver_fcnt = [0] * len(model_v)

for face in model_f:
    v0_idx = face[0] - 1
    v1_idx = face[1] - 1
    v2_idx = face[2] - 1

    v0 = model_v[v0_idx]
    v1 = model_v[v1_idx]
    v2 = model_v[v2_idx]

    face_normal = normal(v0, v1, v2)

    ver_norm[v0_idx] += face_normal
    ver_norm[v1_idx] += face_normal
    ver_norm[v2_idx] += face_normal

    ver_fcnt[v0_idx] += 1
    ver_fcnt[v1_idx] += 1
    ver_fcnt[v2_idx] += 1

def draw_polygon(model_v, model_f, model_ft, model_vt, image):
    light_dir = np.array([0, 0, 1])

    for i in range(len(model_f)):

        face = model_f[i]

        v0_3d = model_v[face[0]-1]
        v1_3d = model_v[face[1]-1]
        v2_3d = model_v[face[2]-1]

        n0 = ver_norm[face[0] - 1]
        n1 = ver_norm[face[1] - 1]
        n2 = ver_norm[face[2] - 1]

        I0 = np.dot(n0, light_dir) / (np.linalg.norm(n0) * np.linalg.norm(light_dir))
        I1 = np.dot(n1, light_dir) / (np.linalg.norm(n1) * np.linalg.norm(light_dir))
        I2 = np.dot(n2, light_dir) / (np.linalg.norm(n2) * np.linalg.norm(light_dir))

        x0_2d = (a_x * v0_3d[0]) / v0_3d[2] + u0
        y0_2d = (a_y * v0_3d[1]) / v0_3d[2] + v00
        z0_3d = v0_3d[2]

        x1_2d = (a_x * v1_3d[0]) / v1_3d[2] + u0
        y1_2d = (a_y * v1_3d[1]) / v1_3d[2] + v00
        z1_3d = v1_3d[2]

        x2_2d = (a_x * v2_3d[0]) / v2_3d[2] + u0
        y2_2d = (a_y * v2_3d[1]) / v2_3d[2] + v00
        z2_3d = v2_3d[2]

        # индексы текстур
        vt0_ind = model_ft[i][0] - 1
        vt1_ind = model_ft[i][1] - 1
        vt2_ind = model_ft[i][2] - 1

        vt0 = model_vt[vt0_ind]
        vt1 = model_vt[vt1_ind]
        vt2 = model_vt[vt2_ind]

        draw_triangle(x0_2d, y0_2d, z0_3d, x1_2d, y1_2d, z1_3d, x2_2d, y2_2d, z2_3d,
                      image, vt0, vt1, vt2, I0, I1, I2, z_buffer)

draw_polygon(model_v, model_f, model_ft, model_vt, img_arr)

img = Image.fromarray(img_arr)
image = ImageOps.flip(img)
image.save("imgtext1.png")
image.show()