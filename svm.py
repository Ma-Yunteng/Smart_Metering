import os
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    fn_dir = "C:/Users/Dushyant Sharma/Desktop/SVM_Dataset/"

    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    c=0
    (images, lables, names, id) = ([], [], {}, 0)
    for (subdir, dirs, files) in os.walk ( fn_dir ):
        for subdir in dirs:
            names[id] = subdir
            mypath = os.path.join ( fn_dir, subdir )
            for item in os.listdir ( mypath ):
                if ".jpg" in item:
                    label = id
                    image = cv2.imread ( os.path.join ( mypath, item ), 0 )
                    r_image = cv2.resize ( image, (30, 30) ).flatten ()
                    if image is not None:
                        images.append ( r_image )
                        lables.append ( names[id] )

            id += 1

    (images, lables) = [np.array ( lis ) for lis in [images, lables]]

    # print ( lables ,"images")
    # print ( images ,"labels")

    nf = 10
    pca = PCA ( n_components=nf )
    #print ( images.shape,"images.shape" )
    pca.fit ( images )
    img_feature = pca.transform ( images )

    #print ( img_feature )

    classifier = SVC ( verbose=0, kernel="poly", degree=3 )
    classifier.fit ( img_feature, lables )
    result=""
    #test_image = cv2.imread("C:/Users/Dushyant Sharma/Desktop/3/",0)
    #test_arr_img = cv2.resize(test_image,(30,30)).flatten()
    for item in os.listdir ( "C:/Users/Dushyant Sharma/Desktop/dataset/Image_Processing/contour/" ):
        if ".jpg" in item:
            label = id
            test_image = cv2.imread ( os.path.join ( "C:/Users/Dushyant Sharma/Desktop/dataset/Image_Processing/contour/", item ), 0 )
            test_arr_img = cv2.resize ( test_image, (30, 30) )
            test_arr_img=np.array(test_arr_img).flatten()

            temp=[]
            temp.append(test_arr_img)
            im_test = pca.transform ( temp )
            pred = classifier.predict ( im_test )
            result = result + pred.item()
    #print(result)

    return(result)

if __name__ == '__main__':
    a=main()
    print(a)
