import cv2
import operator

######################################################################
class ContourWithData ():
    # member variables
    npaContour = None  # contour
    boundingRect = None  # bounding rect for contour
    intRectX = 0  # bounding rect top left corner x location
    intRectY = 0  # bounding rect top left corner y location
    intRectWidth = 0  # bounding rect width
    intRectHeight = 0  # bounding rect height
    fltArea = 0.0  # area of contour
    aspectratio = 0  # aspect ratio of the contour formed
    avgwidth = 0
    avgheight =0

    def Calculate_Parameters(self):  # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight
        self.aspectratio = intWidth / intHeight;

def main(path,img):
    # Read the Image

    frame = cv2.imread(path + img)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    lower = {'red': (166, 84, 141),'orange': (0, 50, 80)}  # assign new item lower['blue'] = (93, 10, 0)
    upper = {'red': (186, 255, 255),'orange': (20, 255, 255)}

    mask = None

    for key, value in upper.items():
        mask = cv2.inRange(hsv, lower[key], upper[key])

    allContoursWithData = []
    mask1=mask.copy()
    contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ValidContour = []

    for i in contours:  # for each contour
        contourWithData = ContourWithData()  # instantiate a contour with data object
        contourWithData.npaContour = i  # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect ( contourWithData.npaContour )  # get the bounding rect
        contourWithData.Calculate_Parameters ()  # get bounding rect info
        contourWithData.fltArea = cv2.contourArea ( contourWithData.npaContour )  # calculate the contour area
        allContoursWithData.append ( contourWithData )  # add contour with data object to list of all contours with data

    for contour in allContoursWithData:
        if contour.fltArea > 0:
            ValidContour.append(contour)

    if len(ValidContour)==0:
        cv2.imwrite(path+"/Image_Processing/significant_img.jpg",frame)
        print("No Insignificant Numbers found.")
        return

    ValidContour.sort(key=operator.attrgetter("intRectX"))
    cv2.drawContours(frame, contours, -1, (54,255,255),5)
    h, w = frame.shape[:2]
    x=None

    if ValidContour[0].intRectX<(w/4):
        for i in range(len(ValidContour)):
            if ValidContour[i].intRectX>w/4:
                x=ValidContour[i].intRectX
    else:
        x=ValidContour[0].intRectX
    initial_y=ValidContour[0].intRectY

    y=initial_y+h

    #v = cv2.line(frame,(x,0),(x,y),(0,255,0),2)

    crop = frame[0:0 + y, 0:0 + x]

   #cv2.imshow ( "Significant_Digits Image", crop )
   # cv2.waitKey ( 0 )

    cv2.imwrite(path+"Image_Processing/significant_img.jpg",crop)
    print ( "Saving Significant Numbers." )
    return

if __name__ == '__main__':
    path= "C:/Users/Dushyant Sharma/Desktop/dataset/"
    main(path,"11.jpg")
    a=cv2.imread(path+"11.jpg")
    b=cv2.imread(path+"Image_Processing/significant_img.jpg")
    cv2.imshow("Original Image",a)
    cv2.imshow("Cropped Image",b)
    cv2.waitKey(0)