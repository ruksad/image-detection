########cheque_tagging.py########
from load_my_model import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_build_image_cheque_categorization_model(
	model_file = 'cheque.h5py')
dlModel= load_build_image_dl_categorization_model('dl_classifier.h5')
def cheque_tagging(input_file):
	output = {}
	img = image.load_img(input_file, target_size=(224, 224))
	x = image.img_to_array(img)
	x = xception.preprocess_input(x)
	x = numpy.array([x])
	x = base_model_Xception.predict(x)
	y_score = model.predict(x)
	prediction = numpy.argmax(y_score)
	score = numpy.max(y_score)
	if prediction > 0:
		output["tag"] = 'cheque'
	else:
		output["tag"] = 'non_cheque'
	output["score"] = score
	return output

def dl_tagging(input_file):
	img = load_img(input_file, target_size=(7, 7))
	img_array = img_to_array(img) #/ 255.0  # Normalize pixel values between 0 and 1
	# img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch
	#
	# # Make predictions
	# prediction = dlModel.predict(img_array)
	#
	# # Convert the prediction to a human-readable class
	# if prediction[0][0] > 0.5:
	# 	return "Driving License"
	# else:
	# 	return "Cheque"

	x = xception.preprocess_input(img_array)
	x = numpy.array([x])
	x = base_model_Xception.predict(x)
	y_score = dlModel.predict(x)
	prediction = numpy.argmax(y_score)
	score = numpy.max(y_score)
	if prediction > 0:
		output["tag"] = 'cheque'
	else:
		output["tag"] = 'non_cheque'
	output["score"] = score
	return output

'''

wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Sample_cheque.jpeg/1200px-Sample_cheque.jpeg


cheque_tagging("1200px-Sample_cheque.jpeg")

{'score': 0.7404399, 'tag': 'cheque'}


wget https://www.fundsindia.com/blog/wp-content/uploads/2017/05/Valid.png

cheque_tagging("Valid.png")

{'score': 0.7928494, 'tag': 'non_cheque'}


wget https://www.ilwindia.com/wp-content/uploads/2019/08/Heriot-Watt-University-Dubai-1.jpg

cheque_tagging("Heriot-Watt-University-Dubai-1.jpg")

{'score': 1.0, 'tag': 'non_cheque'}

'''

########cheque_tagging.py########
