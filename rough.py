"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print(digits.data)
print(digits.target)
print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100)

print(len(digits.data))

x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)
print("Actual value {}".format(digits.target[-1]))
print('Prediction:', clf.predict(digits.data[[-1]]))

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
"""
"""
# import the necessary packages
import numpy as np
import imutils
import cv2
import ContourDetection
from preprocessing import preprocess

preprocess("C:/Users/Dushyant Sharma/Desktop/dataset/","11.jpg")
preprocessed_img = cv2.imread ( "C:/Users/Dushyant Sharma/Desktop/dataset/preprocessed.jpg")
image = cv2.imread ( "C:/Users/Dushyant Sharma/Desktop/5.jpg")
# find all the 'black' shapes in the image
lower = np.array ( [0, 0, 0] )
upper = np.array ( [15, 15, 15] )
shapeMask = cv2.inRange ( image, lower, upper )

# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)

# loop over the contours
for c in cnts:
# draw the contour and show it
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
"""
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread ("C:/Users/Dushyant Sharma/Desktop/5.jpg")
mask = np.zeros ( img.shape[:2], np.uint8 )
bgdModel = np.zeros ( (1, 65), np.float64 )
fgdModel = np.zeros ( (1, 65), np.float64 )
rect = (50, 50, 450, 290)
cv2.grabCut ( img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT )
mask2 = np.where ( (mask == 2) | (mask == 0), 0, 1 ).astype ( 'uint8' )
img = img * mask2[:, :, np.newaxis]
plt.imshow ( img ), plt.colorbar (), plt.show ()
"""
"""
import urllib.request
file= open("Hello.txt","w")
page = urllib.request.urlopen('https://en.wikipedia.org/wiki/The_Power_of_Sympathy')
a=str(page.read())
for i in a:
    file.write(i)
file.close()
file=open("hello.txt","r")
a=file.read()
a.split("\\\\\n")
print(a)
"""
"""
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use ( 'ggplot' )


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure ()
            self.ax = self.fig.add_subplot ( 1, 1, 1 )

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append ( feature )

        self.max_feature_value = max ( all_data )
        self.min_feature_value = min ( all_data )
        all_data = None

        # support vectors yi(xi.w+b) = 1

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array ( [latest_optimum, latest_optimum] )
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange ( -1 * (self.max_feature_value * b_range_multiple),
                                     self.max_feature_value * b_range_multiple,
                                     step * b_multiple ):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # #### add a break here later..
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot ( w_t, xi ) + b) >= 1:
                                    found_option = False
                                    # print(xi,':',yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            opt_dict[np.linalg.norm ( w_t )] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print ( 'Optimized a step.' )
                else:
                    w = w - step

            norms = sorted ( [n for n in opt_dict] )
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print ( xi, ':', yi * (np.dot ( self.w, xi ) + self.b) )

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign ( np.dot ( np.array ( features ), self.w ) + self.b )
        if classification != 0 and self.visualization:
            self.ax.scatter ( features[0], features[1], s=200, marker='*', c=self.colors[classification] )
        return classification

    def visualize(self):
        [[self.ax.scatter ( x[0], x[1], s=100, color=self.colors[i] ) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane ( hyp_x_min, self.w, self.b, 1 )
        psv2 = hyperplane ( hyp_x_max, self.w, self.b, 1 )
        self.ax.plot ( [hyp_x_min, hyp_x_max], [psv1, psv2], 'k' )

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane ( hyp_x_min, self.w, self.b, -1 )
        nsv2 = hyperplane ( hyp_x_max, self.w, self.b, -1 )
        self.ax.plot ( [hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k' )

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane ( hyp_x_min, self.w, self.b, 0 )
        db2 = hyperplane ( hyp_x_max, self.w, self.b, 0 )
        self.ax.plot ( [hyp_x_min, hyp_x_max], [db1, db2], 'y--' )

        plt.show ()


data_dict = {-1: np.array ( [[1, 7],
                             [2, 8],
                             [3, 8], ] ),

             1: np.array ( [[5, 1],
                            [6, -1],
                            [7, 3], ] )}

svm = Support_Vector_Machine ()
svm.fit ( data=data_dict )

predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]

for p in predict_us:
    svm.predict ( p )

svm.visualize ()
"""