#For New SVM (MNIST Dataset)= This Generates classifier
# Import the modules
from mlxtend.data import loadlocal_mnist
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import svm
import numpy as np
from collections import Counter

# Load the data set
#dataset = datasets.fetch_mldata("MNIST Original")


X, y = loadlocal_mnist (
    images_path='C:/Users/Dushyant Sharma/PycharmProjects/Smart_Metering/train-images.idx3-ubyte',
    labels_path='C:/Users/Dushyant Sharma/PycharmProjects/Smart_Metering/train-labels.idx1-ubyte' )


# Extract the features and labels
features = np.array(X, 'int16')
labels = np.array(y, 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print ("Count of digits in dataset", Counter(labels))

# Create an linear SVM object
clf = svm.LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)
