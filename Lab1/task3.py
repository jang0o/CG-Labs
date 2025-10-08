model = []

file = open('model_1.obj')
for l in file:
    l_split = l.split()

    if l_split[0] == 'v':
        model.append(list(map(lambda x: float(x), l_split[1:])))

print(model, 'vaib')