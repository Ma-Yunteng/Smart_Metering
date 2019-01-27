import cv2
import numpy as np
import os
from preprocessing import preprocess
import svm
import ContourDetection
import Crop
from skimage._shared.utils import skimage_deprecation
from sklearn.externals import joblib
from skimage.feature import hog
import warnings

RESIZED_IMAGE_WIDTH = 10
RESIZED_IMAGE_HEIGHT = 10

warnings.filterwarnings ( "ignore", category=skimage_deprecation )


def main():
    #################################   training part    ##################################

    try:
        npaClassifications = np.loadtxt ( "generalresponses.data", np.float32 )  # read in training classifications
    except:
        print ( "error, unable to open classifications (generalresponses.data), exiting program\n" )
        os.system ( "pause" )
        return

    try:
        npaFlattenedImages = np.loadtxt ( "generalsamples.data", np.float32 )  # read in training images
    except:
        print ( "error, unable to open flattened_images (generalsamples.data), exiting program\n" )
        os.system ( "pause" )
        return

    npaClassifications = npaClassifications.reshape ( (npaClassifications.size, 1) )
    # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create ()  # instantiate KNN object

    kNearest.train ( npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications )

    ############################# testing part  ###################################

    path = "C:/Users/Dushyant Sharma/Desktop/dataset/"

    #testing
    initial_image = cv2.imread ( path + image )

    Crop.main ( path, image )
    preprocess ( path, "Image_Processing/significant_img.jpg" )

    a = ContourDetection.main ( path + "Image_Processing/preprocessed.jpg" )
    im = a[0]
    thresh = a[1]
    validContoursWithData = a[2]

    # Load the classifier
    clf = joblib.load ( "digits_cls.pkl" )
    result1 = ""
    result2 = ""

    for i in validContoursWithData:
        thresh1 = thresh.copy ()
        roi = thresh1[i.intRectY:i.intRectY + i.intRectHeight, i.intRectX:i.intRectX + i.intRectWidth]

        # Resize the image
        roi = cv2.resize ( roi, (28, 28), interpolation=cv2.INTER_AREA )
        roi = cv2.dilate ( roi, (3, 3) )

        # Calculate the HOG features
        roi_hog_fd = hog ( roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False )
        nbr = clf.predict ( np.array ( [roi_hog_fd], 'float64' ) )
        result1 = result1 + str ( nbr.item () )

        count = 0
        count = count + 1
        cv2.imwrite ( "C:/Users/Dushyant Sharma/Desktop/dataset/Image_Processing/contour/{}.jpg".format ( count ), roi )

        a = svm.main ()
        result2 = result2 + a

    print ( "The result by New SVM algorithm (By MNIST data-set) is", result1 )
    print ( "The result by SVM algorithm (By Natural Images data-set) is ", result2 )

    strFinalString = ""  # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:  # for each contour
        # draw a green rect around the current char
        cv2.rectangle ( im,  # draw rectangle on original testing image
                        (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                        (contourWithData.intRectX + contourWithData.intRectWidth,
                         contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
                        (0, 255, 0),  # green
                        2 )  # thickness

        x = contourWithData.intRectX
        y = contourWithData.intRectY
        h = contourWithData.intRectHeight
        w = contourWithData.intRectWidth

        roi = thresh[y:y + h, x:x + w]
        # crop char out of threshold image

        imgROIResized = cv2.resize ( roi, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT) )
        # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape ( (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT) )
        # flatten image into 1d numpy array

        npaROIResized = np.float32 ( npaROIResized )
        # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, results, neigh_resp, dists = kNearest.findNearest ( npaROIResized, k=1 )
        # call KNN function find_nearest

        strCurrentChar = str ( int ( (results[0][0]) ) )  # get character from results

        strFinalString = strFinalString + strCurrentChar  # append current char to full string

    print ( "The result by KNN algorithm is " + strFinalString )  # show the full string
    cv2.imshow ( "Original Image", initial_image )
    cv2.imshow ( "imgTestingNumbers", im )  # show input image with green boxes drawn around found digits
    cv2.waitKey ( 0 )  # wait for user key press

    cv2.destroyAllWindows ()  # remove windows from memory

    return

###################################################################################################

if __name__ == "__main__":
    main ()
