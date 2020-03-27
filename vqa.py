
'''导入必要的库与包'''
from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
import spacy, numpy as np
import keras
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from datetime import timedelta
K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')

'''设置路径'''
VQA_weights_file_name   = 'C:/Users/first/Desktop/VQA/models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'C:/Users/first/Desktop/VQA/models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'C:/Users/first/Desktop/VQA/models/CNN/vgg16_weights.h5'

def get_image_model(CNN_weights_file_name):
    
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

'''导入、预处理图片，并进行特征提取'''
def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # 转换每个图片为224×224格式
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    mean_pixel = [103.939, 116.779, 123.68]
    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]
    im = im.transpose((2,0,1)) 
    im = np.expand_dims(im, axis=0) 
    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features

'''导入VQA模型与相应权重'''
def get_VQA_model(VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''
    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

'''对于一个给定的问题，unicode字符串，使用Glove Vector将其转化为向量'''
def get_question_features(question):
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG','bmp'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS
	
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/upload',methods=['POST','GET'])
def upload():
	if request.method == 'POST':
		f = request.files['file']
        # 如果图片不符合规格,提示重新输入
		if not (f and allowed_file(f.filename)): 
			return jsonify({"error":1001,"msg":"请检查上传图片类型，仅限于png\PNG\jpg\JPG\bmp"})
		user_input = request.form.get("name")
		
        # 上传图片到static/images下，并对图片进行处理
        basepath = os.path.dirname(__file__)
		upload_path = os.path.join(basepath,'static/images',secure_filename(f.filename))
		f.save(upload_path)
		img = cv2.imread(upload_path)
		cv2.imwrite(os.path.join(basepath,'static/images','test.jpg'),img)

        # 分别将输入的图片和文字整合到VQA模型中
		image_features = get_image_features('static/images/test.jpg', CNN_weights_file_name)
		question_features = get_question_features(user_input)
		vqa_model = get_VQA_model(VQA_weights_file_name)
		y_output = vqa_model.predict([question_features, image_features])
		y_sort_index = np.argsort(y_output)
		labelencoder = joblib.load(label_encoder_file_name)

        # 输出最有可能的三种结果
		for label in reversed(y_sort_index[0,-1:]):
			data1 = {'nickname1':labelencoder.inverse_transform([label])[0],'number1':round(y_output[0,label]*100,2)}
		for label in reversed(y_sort_index[0,-2:-1]):
			data2 = {'nickname2':labelencoder.inverse_transform([label])[0],'number2':round(y_output[0,label]*100,2)}
		for label in reversed(y_sort_index[0,-3:-2]):
			data3 = {'nickname3':labelencoder.inverse_transform([label])[0],'number3':round(y_output[0,label]*100,2)}
		K.clear_session()
		return render_template('upload_ok.html',userinput=user_input , user1 = data1 , user2 = data2 , user3 = data3 , val1=time.time())
	return render_template('upload.html')
	
if __name__ == '__main__':
	app.run(host='0.0.0.0',port=8987,debug=True)