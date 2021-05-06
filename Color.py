import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass
from statistics import mode

#used to quickly locate the cluster information of a given pixel
@dataclass
class Color:
    r:int
    g:int
    b:int
    group:int


def euclidDistance(x, y):
    distance = np.linalg.norm(x-y)
    return distance


def toRGB(imgArray):
    clusArray = np.empty((len(imgArray), len(imgArray[0])), dtype=object)

    for i in range (len(imgArray)):
        for j in range (len(imgArray[0])):
            temp= Color(imgArray[i][j][0],imgArray[i][j][1], imgArray[i][j][2], -1)
            clusArray[i][j] = temp

    return clusArray


def k_means(img):
    target=[]
    for i in range(5):
        xCord = random.randint(0, len(img[0]) - 1)
        yCord = random.randint(0, len(img) - 1)
        target.append(list(img[yCord][xCord]))
    ctr=0
    ary= toRGB(img)
    while (ctr != 15):
        ctr = 0
        c = [0,0,0,0,0]
        sums = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        for i in range (len(img)):
            for j in range (len(img[0])):
                group = -1
                minNum = 3000
                for k in range(5):
                    cur = euclidDistance(img[i][j], target[k])
                    if cur < minNum:
                        minNum = cur
                        group = k
                ary[i][j].group = group
                sums[group][0] += ary[i][j].r
                sums[group][1] += ary[i][j].g
                sums[group][2] += ary[i][j].b
                c[group]+= 1

        for l in range(5):
            for m in range (3):
                if(c[l] != 0):
                    avg = int(sums[l][m]/c[l])
                else:
                    avg = 0
                if abs(avg - target[l][m]) > 5:
                    target[l][m] = avg
                else:
                    ctr += 1

    return target, ary


def colorLeft(Lside, target, ary):
    for x in range(0, len(Lside)):
        for y in range(0, len(Lside[0])):
            Lside[x][y] = target[ary[x][y].group]

    return Lside


def Gray(img):
    new = np.copy(img)
    img = img.tolist()

    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            new[x][y] = 0.21 * img[x][y][0] + 0.72 * img[x][y][1] + 0.07 * img[x][y][2]
            img[x][y] = 0.21 * img[x][y][0] + 0.72 * img[x][y][1] + 0.07 * img[x][y][2]

    return np.array(img), new


def getPatches(img):
    patch=[]

    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            patch.append((img[i-1:i+2,j-1:j+2],(i,j)))

    return patch


def colorRight(right,left):
    target, cArray = k_means(left)
    finalLefty = np.copy(colorLeft(left, target, cArray))
    gleft = Gray(left)[0]
    gRight, new= Gray(right)
    gLeftPatch=getPatches(gleft)
    counter = 0

    for i in range(1,len(gRight)-1):
        for j in range(1,len(gRight[0])-1):
            # patch=gRight[i-1:i+2,j-1:j+2]
            minNum1,minNum2,minNum3,minNum4,minNum5,minNum6=1000,1000,1000,1000,1000,1000
            patches=[[],[], [], [], [], []]
            samples = random.sample(list(gLeftPatch), 1000)

            for k in samples:
                dist=euclidDistance(k[0],gRight[i-1:i+2,j-1:j+2])
                if dist<minNum1:
                    minNum1=dist
                    patches[1]=patches[0]
                    patches[0]=k[1]
                    continue
                if dist<minNum2:
                    minNum2=dist
                    patches[2]=patches[1]
                    patches[1]=k[1]
                    continue
                if dist<minNum3:
                    minNum3=dist
                    patches[3]=patches[2]
                    patches[2]=k[1]
                    continue
                if dist<minNum4:
                    minNum4=dist
                    patches[4]=patches[3]
                    patches[3]=k[1]
                    continue
                if dist<minNum5:
                    minNum5=dist
                    patches[5]=patches[4]
                    patches[4]=k[1]
                    continue
                if dist<minNum6:
                    minNum6=dist
                    patches[5]=k[1]
                    continue

            for i2 in range(0,len(patches)):
                x=patches[i2][1]
                y=patches[i2][0]
                patches[i2] = cArray[y][x].group

            try:
                freq=mode(patches)
                new[i][j]=target[freq]
            except:
                x=random.randint(0,len(patches)-1)
                draw=patches[x]
                new[i][j]=target[draw]

            counter += 1
        print(counter/(len(gRight)*len(gRight[0]))*100, "%")

    return finalLefty, new


def combine(finalLeft, finalRight):

    finalPicture = []

    for i in range(0, len(finalLeft)):
        finalPicture.append(list(finalLeft[i])+list(finalRight[i]))

    plt.imshow(finalPicture)
    plt.show()

    return finalPicture


def get_left_half(img):
    cropped_img = img[:,:img.shape[1]//2]
    return cropped_img


def get_right_half(img):
    cropped_img = img[:,:img.shape[1]//2:]
    return cropped_img


if __name__ =='__main__':
    inp = str(input())

    img = mpimg.imread('trop2.png')

    half = int(len(img[0])/2)
    left = img[:,:half]
    right = img[:,half:]

    # type s to see the image
    if inp == 's':
        plt.imshow(img)
        plt.show()
    elif inp == 'b':        # Type b to run basic agent
        rightSide= np.copy(right)
        leftSide= np.copy(left)
        finalLeft, finalRight = colorRight(rightSide,leftSide)

        finalPic = combine(finalLeft, finalRight)

        plt.imshow(finalPic)
        plt.show()
