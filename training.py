import numpy as np
import pygame as py
def idx3_to_numpy(file_path):
    with open(file_path, 'rb') as f:
        magic_number = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
        num_images = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
        num_rows = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
        num_cols = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]

        total_pixel = int(num_images * num_rows * num_cols)
        images = np.frombuffer(f.read(total_pixel), dtype=np.uint8).reshape(num_images, num_rows*num_cols)
    return images
def idx1_to_numpy(file_path):
    with open(file_path,"rb") as f:
        magic_number = np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0]
        no_labels = int(np.frombuffer(f.read(4), dtype=np.uint32).byteswap()[0])
        label = np.frombuffer(f.read(no_labels), dtype=np.uint8)
    return label

def numpy_to_binary(np_array,path):

    with open(path,"wb") as f:
        f.write(np_array.tobytes())
def binary_to_numpy(path, dim_2d=False, shape=(),int8=False):
    with open(path,"rb") as f:
        array = np.frombuffer(f.read(),dtype="float")
    if int8:
        with open(path,"rb") as f:
            array = np.frombuffer(f.read(),dtype=np.uint8)
    if dim_2d:
        array = array.reshape(shape)
    return array
def relu(w):
    return np.maximum(0,w)
def softmax(w):
    t = np.exp(w-max(w)).sum()
    return np.exp(w-max(w))/t
def front_propagation(input,w1,w2,b1,b2):
    z1 = np.matmul(w1,input)+b1
    a1 = relu(z1)
    z2 = np.matmul(w2,a1)+b2
    a2 = softmax(z2)
    return a1,z1,a2,z2
def ReLU_deriv(z):
    return z>0

def back_prop(a1,a2,z1,y,x):
    dz2 = (a2 - y)
    dw2 = np.outer(dz2,a1)
    db2 =np.sum(dz2)
    dz1 =np.dot(w2.T,dz2)* ReLU_deriv(z1)
    dz1=dz1.reshape(30,1)
    x=x.reshape(784,1)
    dw1 =np.dot(dz1, x.T)
    db1 =np.sum(dz1)
    return dw1,dw2,db1,db2
def one_to_ten(actual):
    a = np.zeros(10)
    a[actual] = 1
    return a

def loss(predicted,actual):

    a=one_to_ten(actual)
    l = predicted-a
    return (l.dot(l))/2

def update(dw1,dw2,db1,db2,learning_rate=0.0035):
    global w1, w2 , b2 , b1
    w1 = w1-learning_rate*dw1
    w2 = w2-learning_rate*dw2
    b1 = b1-learning_rate*db1
    b2 = b2-learning_rate*db2
    return w1, w2 , b2 , b1
def file_update():
    numpy_to_binary(w1, "weight1")
    numpy_to_binary(w2, "weight2")
    numpy_to_binary(b1, "bias1")
    numpy_to_binary(b2, "bias2")

def verify():
    sum = 0
    test_img=idx3_to_numpy("t10k-images.idx3-ubyte")
    test_label=idx1_to_numpy("t10k-labels.idx1-ubyte")
    for i in range(0,100000):
        r=np.random.randint(0,10000)

        _,_,_,a2=front_propagation(test_img[r], w1, w2, b1, b2)
        a2=np.argmax(a2)
        if a2==test_label[r]:
            sum+=1
    print(sum/1000)

def train(images ,label,start,stop):
    global w1, w2 , b2 , b1
    for i in range(start,stop):
        a1, z1, a2, z2 = front_propagation(images[i], w1, w2, b1, b2)
        dw1, dw2, db1, db2 = back_prop(a1, a2,z1, one_to_ten(label[i]), images[i])
        w1, w2 , b2 , b1 = update(dw1, dw2, db1, db2)
    return w1, w2 , b2 , b1
def reforce_train(images,label):
    global w1, w2 , b2 , b1
    a1, z1, a2, z2 = front_propagation(images, w1, w2, b1, b2)
    dw1, dw2, db1, db2 = back_prop(a1, a2, z1, one_to_ten(label), images)
    w1, w2 , b2 , b1 = update(dw1, dw2, db1, db2)
    return w1, w2 , b2 , b1
def predict(input):
    _,_,_,a2=front_propagation(input,w1,w2,b1,b2)
    return np.argmax(a2)

w1 = binary_to_numpy("weight_one",True,(30, 784))
w2 = binary_to_numpy("weight_two",True,(10, 30))
b1 = binary_to_numpy("bias_one")
b2 = binary_to_numpy("bias_two")


run=True
if __name__ == "__main__" and run:

    py.init()
    hight=140
    widht=140
    s=py.display.set_mode((hight,widht))
    fontObj = py.font.Font(None, 32)
    run=True
    input_size = 140
    output_size = 28
    pic=np.zeros((hight,widht))
    small=np.zeros((28,28))
    while run:
        x,y=py.mouse.get_pos()
        for e in py.event.get():
            if e.type == py.QUIT:
                run=False
            if e.type == py.KEYDOWN:
                if e.key == py.K_SPACE:
                    s.fill((0, 0, 0))
                    numpy_to_binary(small,"small.png")
                    pic = np.zeros((hight, widht))
        if py.mouse.get_pressed()[0]:
            co=np.random.randint(200,255)
            py.draw.circle(s, (co, co, co), (x , y ),5)
            pic=py.surfarray.array_red(s).T
            for i in range(0,20):
                for j in range(0,20):
                    pic[i][j] = 0


        bin_size = input_size // output_size
        small = pic.reshape((1, output_size, bin_size,output_size, bin_size)).max(4).max(2)
        a = predict(small.reshape(784)/255)
        pr = fontObj.render(str(a), True, (255, 255, 255))

        py.draw.rect(s, (0, 0, 0), (0, 0, 20, 20))
        s.blit(pr,(0,0))
        py.display.update()





