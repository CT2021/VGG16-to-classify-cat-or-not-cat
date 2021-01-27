import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
import imutils
image = cv2.imread('./examples/123.jpg')
orig = image.copy()

image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model('./Vgg16_Cats_notCats.model')

(notCats, cats) = model.predict(image)[0]

label = "Cats" if cats > notCats else "Not Cats"
proba = cats if cats > notCats else notCats
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.imwrite('./output.png',output)
cv2.waitKey(0)