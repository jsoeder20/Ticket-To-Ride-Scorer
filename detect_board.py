import cv2 as cv
from matplotlib import pyplot as plt

"""
THIS IS NOT FUNCTIONAL. We tried to use this implementation
adapted from the OpenCV template matching tutorial to automate 
our image cropping process, but we wereunsuccessful. 

We attempted to recognize the corners of the board to pass into a 
perspective cropping function. Unfortunately, this implementation
did come close to recognizing any of the corners of our game 
board. We have left this here for future additions on this project.
"""

img = cv.imread('uncropped_board.jpg', cv.IMREAD_GRAYSCALE)
img2 = img.copy()

temp = cv.imread('bottom_left.jpg', cv.IMREAD_GRAYSCALE)
template = cv.resize(temp, (100, 100))
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    print(top_left[0])
    #print(top_left[1])
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Specify a bright pink color (you can adjust the values for brightness)
    bright_pink = (255, 182, 193)

    # Draw a thick rectangle with bright pink color
    cv.rectangle(img, top_left, bottom_right, color=bright_pink, thickness=10)
    plt.imshow(img)
    
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()