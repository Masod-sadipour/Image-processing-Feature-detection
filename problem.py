import cv2
cv=cv2
import numpy as np

def findCirles(img):
    a, contours, b= cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in contours:

        xs = c[:,0,0]
        ys = c[:,0,1]

        x = int(np.average(xs))
        y = int(np.average(ys))

        r = np.max(xs) - np.min(xs)
        r = int(r/2)

        circles.append([x,y,r])

    return circles

def showCircles(img, circles,name):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i in circles:
        # drawing the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # drawing the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)


    cv2.imshow('detected circles '+name,cimg)
    cv2.waitKey(0)

PairNum = 1


imgA = cv2.imread('data/pair%d/figure_A.bmp'%PairNum,0)
circlesA = findCirles(imgA)
showCircles(imgA, circlesA, "A")

imgB = cv2.imread('data/pair%d/figure_B.bmp'%PairNum,0)

circlesB = findCirles(imgB)
showCircles(imgB, circlesB, "B")

# sorting circles by size

def circleSize(c):
    return -c[2]

circlesA = sorted(circlesA, key=circleSize)
circlesB = sorted(circlesB, key=circleSize)

circlesA = np.array(circlesA)
circlesB = np.array(circlesB)

#number of circles in each figure
print(circlesA.shape)
print(circlesB.shape)

# creating the Text file include XA,YA,RadiusA  and XB,YB,RadiusB
array = np.append(circlesA, circlesB)

max = circlesA.shape[0] if circlesA.shape[0] > circlesB.shape[0] else circlesB.shape[0]

header = "{xA:^5s}\t{yA:^5s}\t{radiusA:^7s}\t\t{xB:^5s}\t{yB:^5s}\t{radiusB:^7s}".format(xA="XA", yA="YA", radiusA="radiusA", xB="XB", yB="YB", radiusB="radiusB")


#Writing the centers and radiuses of the circules in a text file

f = open("problem.txt", "w")
f.write(header + "\n")


for i in range(max):
    if i < circlesA.shape[0]:
        xA = str(circlesA[i][0])
        yA = str(circlesA[i][1])
        radiusA = str(circlesA[i][2])
    else:
        xA = ''
        yA = ''
        radiusA = ''

    if i < circlesB.shape[0]:
        xB = str(circlesB[i][0])
        yB = str(circlesB[i][1])
        radiusB = str(circlesB[i][2])
    else:
        xB = ''
        yB = ''
        radiusB = ''
    out = "{xA:<5s}\t{yA:^5s}\t{radiusA:^7s}\t\t{xB:<5s}\t{yB:<5s}\t{radiusB:^7s}".format(xA=xA, yA=yA, radiusA=radiusA,
                                                                                          xB=xB, yB=yB, radiusB=radiusB)
    f.write(out + "\n")

f.close()
# Calculating the scale

avgA = np.average(circlesA[:,2])
avgB = np.average(circlesB[:,2])
# Z is the scale if there is any.
z = avgB/avgA

# Updating circles of the image A by the calculated scale.
circlesA[:,2] = circlesA[:,2]*z

#Finding the Paires(Some features that are common in both figures for Feature Matching.
# src_pts = Coordinates of the points in the original figure),
#  (dst_pts = Coordinates of the points in the target figure by evaluating radius of the circles)
src_pts = []
dst_pts = []
for p1 in circlesA:
    for p2 in circlesB:
        # print(p1[2])
        if p1[2]>0 and p2[2]>0 and abs(1- p1[2]/p2[2] ) < 0.05:
            src_pts.append(p1[0:2])
            dst_pts.append(p2[0:2])

src_pts=np.float32(src_pts)
dst_pts=np.float32(dst_pts)


# Find the transformation between points, standard RANSAC
transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

#printing and saving the transformation_matrix
print(transformation_matrix)
np.savetxt('Problem3a.txt',transformation_matrix, header='transformation_matrix',fmt='%.1e',delimiter=',')


#  Just for checking the code we add a "Gray Ring" to the figure A
cv2.circle(imgA,(20,100),10,127,4)
# apply transformation

imgAT = cv.warpPerspective(	imgA, transformation_matrix, (imgA.shape[1],imgA.shape[0]))


# showing imageA, imageA', image B.
cv2.imshow('imgA', imgA)

cv2.imshow('imgAT', imgAT)
cv2.imwrite('imgAT.bmp',imgAT)

cv2.imshow('imgB', imgB)
cv2.waitKey(0)
