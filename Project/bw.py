import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = [[(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)) for i in range(96)] for j in range(96)]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



ret = rgb2gray(img)
print(ret)
