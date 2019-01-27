## Copyright to rishabh shukla #theboss __ keep away

import cv2;
import numpy as np;

def preprocess(path,name):

    print("Preprocessing Starts!!")
    # Read the Image
    image = cv2.imread(path +  name)

    MSF = cv2.pyrMeanShiftFiltering ( image, 21, 51 )

    # convert to grayscale
    gray_image = cv2.cvtColor ( MSF, cv2.COLOR_BGR2GRAY )


    # # blur it
    # blurred_image = cv2.GaussianBlur ( gray_image, (7, 7), 0 )

    # Otsu's Binarization
    a, binary_image = cv2.threshold ( gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # Run the Canny edge detector
    canny = cv2.Canny ( binary_image, 100, 150 )

    # Converting to bgr
    gray_to_bgr = cv2.cvtColor ( binary_image, cv2.COLOR_GRAY2BGR )

    # converting to hsv
    hsv = cv2.cvtColor ( gray_to_bgr, cv2.COLOR_BGR2HSV )

    # using HSL to mark white
    sensitivity = 100
    lower_white = np.array ( [0, 0, 255 - sensitivity] )
    upper_white = np.array ( [255, sensitivity, 255] )
    # black background pe White text
    mask = cv2.inRange ( hsv, lower_white, upper_white )

    # Converting Black And white
    mask = 255 - mask

    # Writing Masked Image
    cv2.imwrite ( path + "/Image_Processing/preprocessed.jpg", mask )
    print("Preprocessing Complete!!")

    return

if __name__ == '__main__':
    path = "C:/Users/Dushyant Sharma/Desktop/dataset/"
    preprocess(path,"11.jpg")
    a=cv2.imread(path+"11.jpg")
    b=cv2.imread(path+"/Image_Processing/preprocessed.jpg")
    cv2.imshow("Original Image", a)
    cv2.imshow("Pre-processed Image", b)
    cv2.waitKey(0)
