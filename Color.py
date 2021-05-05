import time
import numpy as np
import random
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
from statistics import mode
import time

#used to quickly locate the group information of a given pixel
@dataclass
class Color():
    r:int
    g:int
    b:int
    group:int


#reColor the left image in terms of representative Colors
def reColorLeft(Lside, target, ary):
    for y in range(0, len(Lside)):
        for x in range(0, len(Lside[0])):
            Lside[y][x] = target[ary[y][x].group]
    return Lside

#a method to turn an image to greyscale
def toGray(image):
    #reColor each pixel
    new = np.new(image)
    image = image.tolist()
    for i in range(0,len(image)):
        for j in range(0, len(image[i])):
            image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
            new[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]
    return np.array(image), new

#calculate the euclidean distance
def Edistance(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

#turn the image format to [r,g,b,group] for later convenience
def format(img):
    ary = np.empty((len(img), len(img[0])), dtype=object)
    for i in range (len(img)):
        for j in range (len(img[0])):
            temp= Color(img[i][j][0],img[i][j][1], img[i][j][2], -1)
            ary[i][j] = temp
    return (ary)





#the kmeans algorithm to find the target of our image data
def kmeans(img):
    target=[]
    #generates 5 random points
    for i in range(5):
        random_y = random.randint(0, len(img) - 1)
        random_x = random.randint(0, len(img[0]) - 1)
        target.append(list(img[random_y][random_x]))

    ctr=0

    ary= format(img)
    while (ctr!=15):
        sum=[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        counter=[0,0,0,0,0]
        ctr=0

        # goes through every pixel
        for y in range (len(img)):
            for x in range (len(img[0])):
                #print("x", x, "y", y)
                shortestDist=3000
                group=-1

                for j in range(5):
                    # finding the closest centroid
                    newDist = Edistance(img[y][x], target[j])
                    if newDist < shortestDist:
                        shortestDist=newDist
                        group=j

                #tag the pixel
                ary[y][x].group= group
                #add to the array of sums
                sum[group][0] +=ary[y][x].r
                sum[group][1] +=ary[y][x].g
                sum[group][2] +=ary[y][x].b
                counter[group]+= 1
        #finds the average
        for k in range(5):
            for l in range (3):
                if(counter[k]!= 0):
                    avg= int(sum[k][l]/counter[k])
                else:
                    avg= 0
                if abs(avg-target[k][l]) > 5:
                    target[k][l] = avg
                else:
                    ctr += 1


    return target, ary



#get all of the 3x3 patches in an image
def get_patches(img):
    patches=[]
    #iterate through grayleft
    #iterate through rows
    for i in range(1,len(img)-1):
        #iterate through columns
        for j in range(1,len(img[0])-1):
            #grayleft[i][j] starts on middle pixel
            #find the rest of the patch (adjacent pixels)
            patches.append((img[i-1:i+2,j-1:j+2],(i,j)))
    return patches

#reColor the right side by finding 6 most similar in testing data
def reColor_right(right,left):
    target, ary = kmeans(left)
    #FINAL OUTPUT FOR LEFT (representative Colors)
    final_left = np.new(reColorLeft(left, target, ary))

    grayleft = toGray(left)[0]

    grayright, new= toGray(right)

    #plt.imshow(grayleft)
    #plt.show()

    grayleftPatch = get_patches(grayleft)

    tracker = 0

    #iterate through testing
    #iterate through rows
    for i in range(1,len(grayright)-1):
        #iterate through columns
        for j in range(1,len(grayright[0])-1):
            patch=grayright[i-1:i+2,j-1:j+2]
            min1,min2,min3,min4,min5,min6=1000,1000,1000,1000,1000,1000
            pixels=[[],[], [], [], [], []]
            #find six patches

            #take a sample from the total training data to compare with test data
            #the higher the number the better the resulting image quality
            samples = random.sample(list(grayleftPatch), 1000)

            for k in samples:
                dist=Edistance(k[0],grayright[i-1:i+2,j-1:j+2])
                if dist<min1:
                    min1=dist
                    pixels[1]=pixels[0]
                    pixels[0]=k[1]
                    continue
                if dist<min2:
                    min2=dist
                    pixels[2]=pixels[1]
                    pixels[1]=k[1]
                    continue
                if dist<min3:
                    min3=dist
                    pixels[3]=pixels[2]
                    pixels[2]=k[1]
                    continue
                if dist<min4:
                    min4=dist
                    pixels[4]=pixels[3]
                    pixels[3]=k[1]
                    continue
                if dist<min5:
                    min5=dist
                    pixels[5]=pixels[4]
                    pixels[4]=k[1]
                    continue
                if dist<min6:
                    min6=dist
                    pixels[5]=k[1]
                    continue

                #get Color of 6 middel pixels
            for l in range(0,len(pixels)):
                x=pixels[l][1]
                y=pixels[l][0]

                #replace the patches/coordinates we got with the Colors they represent
                pixels[l] = ary[y][x].group

            try:
                mostFrequent=mode(pixels)
                new[i][j]=target[mostFrequent]
            except:
                x=random.randint(0,len(pixels)-1)
                tie=pixels[x]
                new[i][j]=target[tie]

            tracker += 1
        print(tracker/(len(grayright)*len(grayright[0]))*100, "%")

    plt.imshow(final_left)
    plt.show()

    plt.imshow(new)
    plt.show()
    return final_left, new

#combine two pictures into one
def combinePic(final_left, new):
    new = []
    for i in range(0, len(final_left)):
        new.append(list(final_left[i])+list(new[i]))

    plt.imshow(new)
    plt.show()
    return new
