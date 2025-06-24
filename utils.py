import numpy as np 
import cv2
from skimage.feature import hog
import joblib
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from skimage.feature import hog


SVM_Model = joblib.load('C:/Users/gabyl/OneDrive/Documentos/Gaby/University/Semester 8/Data Science and Big Data in the Industry/Project/Project/svm_model.joblib')

def load_images_and_mask(image_path,mask_path,target_size = (400,400)) : 
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
    
    image = cv2.resize(image,target_size)
    mask = cv2.resize(mask,target_size)
    
    return image,mask

def load_images_NormalImage(image_path,target_size = (400,400)) : 
    
    image = cv2.imread(image_path)
    
    image = cv2.resize(image,target_size)
    print(image)

    
    return 

def hog_extractor(image,mask) : 
    
    masked_image = cv2.bitwise_and(image,image,mask=mask)
    
    gray_image = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
    
    features,hot_image = hog(gray_image,pixels_per_cell=(16,16),cells_per_block=(2,2),visualize=True)
    
    return features

def hog_extractor_NormalImage(image) : 
    
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    features,hot_image = hog(gray_image,pixels_per_cell=(16,16),cells_per_block=(2,2),visualize=True)
    
    return features


def classify_single_ultrasound_image(image_path, mask_path=None, target_size=(400, 400)):
    """
    Classifies a single ultrasound image using the trained SVM model.
    
    Parameters:
    - image_path: Path to the image (.png)
    - mask_path: Path to the mask (.png) if available; if None, it's treated as a normal image
    - target_size: Tuple for resizing (default (400, 400))
    
    Returns:
    - predicted class label
    """
    if mask_path:
        # If mask is provided, it's a benign or malignant image
        image, mask = load_images_and_mask(image_path, mask_path, target_size)
        features = hog_extractor(image, mask)
    else:
        # Normal image or external image without mask
        image = load_images_NormalImage(image_path, target_size)
        features = hog_extractor_NormalImage(image)

    features = np.array(features).reshape(1, -1)
    prediction = SVM_Model.predict(features)
    return prediction[0]