import numpy as np
import training

images = training.idx3_to_numpy("train-images.idx3-ubyte").reshape(60000,28,28)
im=np.zeros((60000,28,28),dtype=np.uint8)

im =np.roll(images,3,axis=2)
print(im.shape)
training.numpy_to_binary(im,"mnist_right")