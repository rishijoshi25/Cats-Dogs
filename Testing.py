from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from imutils import paths
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of 'bins' per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

    # images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
    # channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
    # mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
    # histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
    # ranges : this is our RANGE. Normally, it is [0,256].

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]) #the last parameter is HSV hue range

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
 
    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d1", "--dataset1", required=True,
	help="path to input dataset")
ap.add_argument("-d2", "--dataset2", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("Processing the images...")
imagePaths1 = list(paths.list_images(args["dataset1"]))
imagePaths2 = list(paths.list_images(args["dataset2"]))

# initialize the raw pixel intensities matrix, the features/dimensions matrix
# and label list
rawImages = [] # store raw image pixel intensities, 3072-d feature vector, 3072 because 32 x 32 = 1024px x 3 (rgb) = 3072 dimensions for each image
features = []   # store histagram features, the 512-d feature vector
labels = []     # class labels i.es "cat" or "dog"

# loop over input images
for(i, imagePath) in enumerate(imagePaths1):
    # load the image and extract the class label (assuming the path as the format
    # /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    # update the raw images, features, and labels matrices respectively
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths1)))

testImages = []
testFeatures = []
for(i, imagePath) in enumerate(imagePaths2):
    # load the image and extract the class label (assuming the path as the format
    # /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    #label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    # update the raw images, features, and labels matrices respectively
    testImages.append(pixels)
    testFeatures.append(hist)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths2)))

testImages = np.array(testImages)
testFeatures = np.array(testFeatures)

rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
#indices = range(25000)
#indice = np.array(indices)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

print("[INFO] pixels matrix: {:.2f}MB".format(
	testImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	testFeatures.nbytes / (1024 * 1000.0)))

# partition the data into training and testing splits, using 75%
# of data for training and the remaining 25% for testing
#(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
#(trainFeat, testFeat, trainLabels, testLabels, indices_train, indices_test) = train_test_split(features, labels, indice, test_size=0.25, random_state=42)

# construct the set of hyperparameters to tune
#params = {"n_neighbors": np.arange(1, 31, 2),
#	"metric": ["euclidean", "cityblock"]}



# train and evaluate a K-NN classifier on the raw pixel intensities
#print("Evaluating raw pixel accuracy.....!")
#model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
#model.fit(trainRI, trainRL)
#acc = model.score(testRI, testRL)
#print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a K-NN classifier on the histogram representations
print("Evaluating histogram....")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(features, labels)
#acc = model.score(testFeatures, labels)
# tune the hyperparameters via a cross-validated grid search
#print("[INFO] tuning hyperparameters via randomized search")
#model = KNeighborsClassifier(n_jobs=args["jobs"])
#rv = RandomizedSearchCV(model, params)
#rv.fit(trainFeat, trainLabels)
#acc = rv.score(testFeat, testLabels)
#print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

print("\n------Prediction-----\n")
#prediction = rv.predict(testFeat)
prediction = model.predict(testFeatures)

k = 0
for i in range(10):
    print("The prediction: {}".format(prediction[k]))
    image = cv2.imread(imagePaths2[k])
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    k = k + 1
    if k>10:
        break








    
