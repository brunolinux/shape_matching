from PIL import Image

# 逆时针角度为正
# 顺时针 2.15 开始, 隔 5 度 旋转一次
angle = -2.15
step = -5

index = 0
img = Image.open("../img/circle_train.bmp")

while angle > -90:
    img.rotate(angle).save("../img/circle_rotate_{}.bmp".format(index))
    index += 1
    angle += step




