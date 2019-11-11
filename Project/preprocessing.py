import numpy as np

def main(inputArray):
    processedImage = inputArray.copy()
    for i in range(10,len(inputArray-10)):
        for j in range(13,len(inputArray[i])-13):
            
            r = int(inputArray[i][j][0])
            g = int(inputArray[i][j][1])
            b = int(inputArray[i][j][2])

            rgb = r + g + b

          
            if(rgb < 325):
              processedImage[i][j] = [0,0,0]
            else:
              processedImage[i][j] = [255,255,255]


    np.save("imageData",processedImage)
#    np.save("imageData",inputArray)
