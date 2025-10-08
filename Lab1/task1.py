from PIL import Image
import numpy as np

# # img_matr = np.zeros((600, 800), dtype=np.uint8) # ДЛЯ ЧЕРНОГО ИЗОБРАЖЕНИЯ
# img_matr = np.ones((600, 800), dtype=np.uint8) * 255
# img = Image.fromarray(img_matr, "L")
# # img.save("1.png") # ДЛЯ ЧЕРНОГО ИЗОБРАЖЕНИЯ
# img.save("2.png") # ДЛЯ БЕЛОГО ИЗОБРАЖЕНИЯ
# img.show()

img_matr = np.zeros((600,800,3), dtype=np.uint8)
for i in range(600):
    for j in range(800):
        img_matr[i,j] = [255,255,0]

img = Image.fromarray(img_matr, mode="RGB")
img.save("img1.png")
img.show()