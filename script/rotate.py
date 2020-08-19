from PIL import Image

# 逆时针角度为正
angle = -2.15
step = -5

index = 0
img = Image.open("../img/circle_train.bmp")

while angle > -90:
    img.rotate(angle).save("../img/circle_rotate_{}.bmp".format(index))
    index += 1
    angle += step




