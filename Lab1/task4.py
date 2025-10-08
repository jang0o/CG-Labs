import numpy as np
from PIL import Image, ImageOps

model = []

file = open('model_1.obj')
for l in file:
    l_split = l.split()

    if l_split[0] == 'v':
        model.append(list(map(lambda x: float(x), l_split[1:])))


height, width = 1000,1000

img_matr = np.zeros((height, width), dtype=np.uint8)
for v in model:
    x = round(v[0]*5000+500)
    y = round(v[1]*5000+500)
    img_matr[y][ x] = 255


img = Image.fromarray(img_matr, "L")
img = ImageOps.flip(img)
img.save("img2.png")
img.show()
# print(model, 'vaib')