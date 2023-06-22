import numpy as np
import pandas as pd
import cv2 as cv
from skimage.feature import graycomatrix, graycoprops
import joblib

indextable = ['dissimilarity', 'contrast', 'homogeneity', 'energy','ASM', 'correlation', 'Label']
width, height = 400, 400
distance = 10
teta = 90


def get_feature(matrix, name):
    feature = graycoprops(matrix, name)
    result = np.average(feature)
    return result

def preprocessingImage(image):
    test_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    test_img_gray = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img_thresh = cv.adaptiveThreshold(test_img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,3)
    cnts = cv.findContours(test_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        test_img_ROI = test_img[y:y+h, x:x+w]
        break
    test_img_ROI_resize = cv.resize(test_img_ROI, (width, height))
    test_img_ROI_resize_gray = cv.cvtColor(test_img_ROI_resize, cv.COLOR_RGB2GRAY)
    
    return test_img_ROI_resize_gray    

def extract(path):
    data_eye = np.zeros((6, 1))
    
    image = cv.imread(path)
    img = preprocessingImage(image)
    
    glcm = graycomatrix(img, [distance], [teta], levels=256, symmetric=True, normed=True)
    
    for i in range(len(indextable[:-1])):
        features = []
        feature = get_feature(glcm, indextable[i])
        features.append(feature)
        data_eye[i, 0] = features[0]
    return pd.DataFrame(np.transpose(data_eye), columns=indextable[:-1])

obj = {
    0.0: "Normal",
    1.0: "Cataract"
}
check = []
def predict(path):
    model_rfc = joblib.load("rfc1.pkl")
    model_knn = joblib.load("knn1.pkl")
    model_lr = joblib.load("lr1.pkl")
    model_nb = joblib.load("nb1.pkl")
    model_svm = joblib.load("svm1.pkl")
    X = extract(path)
    results = []
    results.append(obj[model_rfc.predict(X)[0]])
    results.append(obj[model_knn.predict(X)[0]])
    results.append(obj[model_lr.predict(X)[0]])
    results.append(obj[model_nb.predict(X)[0]])
    results.append(obj[model_svm.predict(X)[0]])
    normal_count = 0
    cataract_count = 0
    for result in results:
        if (result == 'Normal'):
            normal_count += 1
        else:
            cataract_count+=1
    if(normal_count > cataract_count):
        actual_result = "Normal"
    else:
        actual_result = "Cataract"            
    check.append(actual_result)
    # print(results)
    # print('\n\n\n\n\nThe Predicted Output for image {} is {}\n\n\n\n\n'.format(path,actual_result))

# print("***************   The prediction for Normal Image     ****************")
# predict(f'Images/new_normal/0111.jpg')
# print("----------------------------------------------------------------------\n\n")
# print("***************   The prediction for Cataract Image   ****************")
# predict(f'Images/new_cataract_copy/0831.jpg') 

for i in range(1,1375):
    predict(f'Images/new_normal/{str(i).zfill(4)}.jpg')

normal = 0
cataract = 0
for i in check:
    if( i == "Normal"):
        normal += 1
    else:
        cataract += 1
print("The output for Normal images is")
print("\n\n\nThe Normal count is "+str(normal))        
print("\n\n\nThe Cataract count is "+str(cataract))        

    
