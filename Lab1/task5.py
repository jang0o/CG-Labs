import numpy as np

model_v = []
model_f = []

file = open('model_1.obj')
for l in file:
    l_split = l.split()

    if l_split[0] == 'v':
        model_v.append([float(l_split[1]), float(l_split[2]), float(l_split[3])])
    if l_split[0] == 'f':
        model_f.append([int(l_split[1].split('/')[0]), int(l_split[2].split('/')[0]), int(l_split[3].split('/')[0])])
    # if i with you i wanna be me too i wanna be me too


