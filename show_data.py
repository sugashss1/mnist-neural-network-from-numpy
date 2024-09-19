import numpy as np
import pygame as py
import sys
import training
def draw_from_numpy(pos,value):
    x, y = pos
    py.draw.rect(s,(value,value,value),(x*15,y*15,x*15+15,y*15+15))

images = training.idx3_to_numpy("train-images.idx3-ubyte").reshape(60000,28,28)
with open("small.png","rb") as f:
    img = (np.frombuffer(f.read(),dtype=np.uint8).reshape(28,28))


label = training.idx1_to_numpy("train-labels.idx1-ubyte")
print(label)
print(images.shape)
py.init()
s=py.display.set_mode((420,420))
no_img=0
fontObj = py.font.Font(None, 32)
while True:
    la = fontObj.render(str(label[no_img]), True, (255, 255, 255))
    p=training.predict(images[no_img].reshape(784))
    pa= fontObj.render(str(label[no_img]), True, (255, 255, 255))
    for e in py.event.get():
        if e.type == py.QUIT:
            sys.exit()
        if e.type == py.MOUSEBUTTONDOWN:
            no_img +=1
        if e.type == py.KEYDOWN:
            no_img-=1
    for i in range(0,28):
        for j in range(0, 28):
            draw_from_numpy((i,j),img[j][i])
    s.blit(la,(0,0))
    s.blit(pa,(0,400))
    py.display.update()