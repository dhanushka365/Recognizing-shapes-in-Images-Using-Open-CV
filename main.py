#import libraries
import cv2
import numpy as np
import os


#image filenames
files = ('subway.jpg', 'breakfast.jpg', 'dinner.jpg', 'building.jpg',)
f = os.path.join('images', files[0])

#define a function for viewing images

def view_image(i):
    cv2.imshow('view', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#read an image from file

i = cv2.imread(f)
view_image(i)

#imspect image content

print(i.shape)
print(i[0, 0, :])

#gray-scale
i_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
print(i_gray.shape)
print(i_gray[0, 0])
view_image(i_gray)

#X gradient

soblex = cv2.Sobel(i_gray, cv2.CV_64F, 1, 0)
abs_soblex = np.absolute(soblex)
view_image(abs_soblex/np.max(abs_soblex))

# y gradient,change direction
sobley = cv2.Sobel(i_gray, cv2.CV_64F, 0, 1)
abs_sobley = np.absolute(sobley)
view_image(abs_sobley/np.max(abs_sobley))


# Magnitude of gradient vectors
magnitude = np.sqrt(soblex**2 + sobley**2)
view_image(magnitude / np.max(magnitude))

#canny edge detection
edges = cv2.Canny(i_gray, 200, 250)
view_image(edges)

#hough transform for lines

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=1. * np.pi/180.0,
    threshold=20,
    minLineLength=25,
    maxLineGap=5,
)
i_lines = i.copy()
for l in lines:
    x1,y1,x2,y2 = l[0]
    cv2.line(i_lines, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
view_image(i_lines)

#hough transform for circle
circles = cv2.HoughCircles(
    i_gray,
    method=cv2.HOUGH_GRADIENT,
    dp=2,
    minDist=35,
    param1=150,
    param2=40,
    minRadius=15,
    maxRadius=25
)

i_circles = i.copy()
for x, y, r in circles[0]:
    cv2.circle(
        i_circles,
        (int(x), int(y)),
        int(r),
        (0, 0, 255),
        thickness=3
    )

view_image(i_circles)

#Blur the image first

i_blurred = cv2.GaussianBlur(
    i_gray,
    ksize=(21,21),
    sigmaX=0

)
view_image(i_blurred)

#circle detection on blurred image

circles = cv2.HoughCircles(
    i_blurred,
    method=cv2.HOUGH_GRADIENT,
    dp=2,
    minDist=35,
    param1=150,
    param2=40,
    minRadius=15,
    maxRadius=25
)

i_circles = i.copy()
for x, y, r in circles[0]:
    cv2.circle(
        i_circles,
        (int(x), int(y)),
        int(r),
        (0, 0, 255),
        thickness=3
    )

view_image(i_circles)