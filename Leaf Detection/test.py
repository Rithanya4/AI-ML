import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import cv2


json_file = open('model1.json' ,'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model1.h5")
print("Loaded model from disk")

label=["Apple_black_rot", "Apple_leaf_healthy", "Apple_scab_leaf"]

test_image = image.load_img(r'C:\Users\rithu\AppData\Local\Programs\Python\Python39\LeafDiseaseDetection\dataset\test\Apple_black_rot_leaf_4.jpeg', target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image  = np.expand_dims(test_image , axis = 0)
result = model.predict(test_image)
print(result)
fresult = np.max(result) #takes max of predicted pbls
label2 = label[result.argmax()]
print(label2)
