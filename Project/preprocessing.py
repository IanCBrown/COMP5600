import numpy as np

#takes 3d array as input and converts to a 2d array in black/white

def main(inputArray):
    processedImage = np.zeros((int(len(inputArray)-10),int(len(inputArray[0])-13)))
    for i in range(10,len(inputArray)-10):
        for j in range(13,len(inputArray[i])-13):
            
            r = int(inputArray[i][j][0])
            g = int(inputArray[i][j][1])
            b = int(inputArray[i][j][2])

            rgb = r + g + b

            if(rgb < 325):
              processedImage[i][j] = 0
            else:
              processedImage[i][j] = 255

            processedImage = processedImage

    np.save("imageData",processedImage)
#    np.save("imageData",inputArray)
