import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_thresh, upper_thresh):
    return cv2.Canny(img, low_thresh, upper_thresh)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1),(x2,y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # drawing hough lines

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., y=0.):
    # initial_img = original image,  img = output of hough lines

    return cv2.addWeighted(initial_img, a, img, b, y)


if __name__ == '__main__':
    
    cap = cv2.VideoCapture('test_videos/solidWhiteRight.mp4')
    
    while True:
        _,image = cap.read() 

        grayscaled = grayscale(image)
            
        blured = gaussian_blur(grayscaled, 5)

        edge = canny(blured, 50, 150)
        
        imshape = image.shape
        vertices = np.array([[(0, imshape[0]), (460,322), (520,322), (imshape[1], imshape[0])]],
                                dtype = np.int32)
        roi_image = region_of_interest(edge, vertices)

        kernel = np.ones((5,5), np.uint8)
        roi_image = cv2.dilate(roi_image, kernel, iterations=1)
        drawed_line = hough_lines(roi_image, 2, np.pi/180,150, 90,  80)
       
        final = weighted_img(drawed_line, image)
        cv2.imshow("roi1", cv2.cvtColor(final,cv2.COLOR_BGR2RGB)) 


        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

