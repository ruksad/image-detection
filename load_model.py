import time
import numpy
import hashlib
from PIL import *
from keras.utils import *
from keras.losses import *
from keras.layers import *
from keras.metrics import *

#from jessica_local_spark_building import sqlContext
from pyspark.sql.types import StructType, StructField, StringType

from pyspark import StorageLevel

from keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import xception

base_model_Xception = xception.Xception(weights='xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

def build_image_categorization_model(gpus = None):
	model = Sequential()
	model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	if gpus is not None:
		model = multi_gpu_model(model, gpus = gpus)
	return model

def train_image_categorization_model(
	x_npy, y_npy,
	x_document_id_npy,
	gpus = None,
	epochs = 3,
	positive_weight = 1,
	batch_size = 512,
	model_file = None,
	output_prediction_json = None):
	#####
	print('load data and label from npy files')
	x = numpy.load(x_npy)
	x_document_id = numpy.load(x_document_id_npy)
	y = numpy.load(y_npy)
	####
	print('building model')
	model = build_image_categorization_model(gpus = gpus)
	model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop', 
		metrics=['accuracy'])
	print('training the model')
	model.fit(x, y, 
		batch_size=batch_size, 
		epochs=epochs,
		class_weight = {1:positive_weight, 0:1})
	if model_file is not None:
		print('saving the model')
		model.save_weights(model_file)
	#####
	print('predicting the labels from the trained model')
	y_score = model.predict(x)
	label_predicted = numpy.argmax(y_score,axis=-1)
	label = numpy.argmax(y,axis=-1)
	label_confidence = numpy.max(y_score,axis=1)
	print('building the dataframe of the prediciton results')
	data = [(str(d), int(l), int(p), float(s)) 
		for d, l, p, s in zip(x_document_id,
		label,
		label_predicted,
		label_confidence)]
	###
	df_prediction = sqlContext.createDataFrame(data, 
	['document_id', 'label', 'prediction', 'score']).persist(StorageLevel.MEMORY_AND_DISK)
	####
	if output_prediction_json is not None:
		print('saving the prediction results')
		df_prediction.write.mode('Overwrite').json(output_prediction_json)
	#####
	df_prediction.registerTempTable('df_prediction')
	sqlContext.sql(u"""
		SELECT label, prediction, COUNT(*)
		FROM df_prediction
		GROUP BY label, prediction
		""").show()
	return model

def load_build_image_cheque_categorization_model(
	model_file,
	gpus = None):
	model = build_image_categorization_model(gpus = gpus)
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop', 
		metrics=['accuracy'])
	#model._make_predict_function()
	return model

def load_build_image_dl_categorization_model(
		model_file,
		gpus = None):
	model = build_image_categorization_model(gpus = gpus)
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])
	#model._make_predict_function()
	return model
def image_tagging(
	x, model,
	tag_name):
	output = {}
	x = xception.preprocess_input(x)
	x = numpy.array([x])
	x = base_model_Xception.predict(x)
	y_score = model.predict(x)
	prediction = numpy.argmax(y_score)
	score = numpy.max(y_score)
	if prediction > 0:
		output["tag"] = tag_name
		output["score"] = score
	return output


