import cv2
import operator
import statistics as st

allContour_height_list=[]
allContour_width_list =[]
slag_height=10
slag_width = 20

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

    def checkIfContourIsValid(self):  # this is oversimplified, for a production grade program
        if self.aspectratio > 1.2:
            return False
        elif self.fltArea < (self.avgheight* self.avgwidth)+100 and self.fltArea < 300 :
            return False  # much better validity checking would be necessary
        return True

    def HeightContourValidation(self):
        avgheight = st.mean(allContour_height_list)
        if self.intRectHeight > (avgheight+slag_height) or  self.intRectHeight < (avgheight-slag_height) :
            return False
        return True

    def WidthContourValidation(self):
        avgwidth = st.mean(allContour_width_list)
        if self.intRectWidth > (avgwidth+slag_width) or  self.intRectWidth < (avgwidth-slag_width) :
            return False
        return True

    def Contour_heightandwidth_list(self):
        allContour_height_list.append(self.intRectHeight)
        allContour_width_list.append(self.intRectWidth)

####################################################################################

def main(preprocessed_path):

    print("Contour Detection Starts")
    im = cv2.resize(cv2.imread(preprocessed_path), (350, 100),cv2.INTER_AREA )
  #  im=cv2.resize ( im, (400, 200) )
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur ( gray, (5, 5), 0 )  # blur

    # filter image from grayscale to black and white
    thresh = cv2.adaptiveThreshold ( imgBlurred,  # input image
                                 255,  # make pixels that pass the threshold full white
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 # use gaussian rather than mean, seems to give better results
                                 cv2.THRESH_BINARY_INV,
                                 # invert so foreground will be white, background will be black
                                 11,  # size of a pixel neighborhood used to calculate threshold value
                                 2 )  # constant subtracted from the mean or weighted mean

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    allContoursWithData = []  # declare empty lists,
    validContoursWithData = []  # we will fill these shortly

    for i in contours:  # for each contour
        contourWithData = ContourWithData ()  # instantiate a contour with data object
        contourWithData.npaContour = i  # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect ( contourWithData.npaContour )  # get the bounding rect
        contourWithData.Calculate_Parameters ()  # get bounding rect info
        contourWithData.fltArea = cv2.contourArea ( contourWithData.npaContour )  # calculate the contour area
        allContoursWithData.append ( contourWithData )  # add contour with data object to list of all contours with data

    temp = []
    for contourWithData in allContoursWithData:  # for all contours
        if contourWithData.checkIfContourIsValid ():  # and contourWithData.HeightContourValidation ():  # check if valid
            temp.append ( contourWithData )  # if so, append to valid contour list

    for contourWithData in temp:
        contourWithData.Contour_heightandwidth_list ()

    for i in temp:
        if i.HeightContourValidation () and i.WidthContourValidation():
            validContoursWithData.append ( i )

    # print ( allContour_height_list )
    # print ( "The mean height is {}".format ( st.mean ( allContour_height_list ) ) )
    # print ( allContour_width_list )
    # print ( "The mean width is {}".format ( st.mean ( allContour_width_list ) ) )

    validContoursWithData.sort ( key=operator.attrgetter ( "intRectX" ) )  # sort contours from left to right

    for contourWithData in validContoursWithData:  # for each contour
        # draw a green rect around the current char
        cv2.rectangle ( im,  # draw rectangle on original testing image
                        (contourWithData.intRectX, contourWithData.intRectY),  # upper left corner
                        (contourWithData.intRectX + contourWithData.intRectWidth,
                         contourWithData.intRectY + contourWithData.intRectHeight),  # lower right corner
                        (0, 255, 0),  # green
                        2 )  # thickness
    print("Contours Detected Successfully")
    return [im,thresh,validContoursWithData]
