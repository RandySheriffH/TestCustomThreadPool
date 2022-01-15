import cv2
from onnx import *
from onnx import numpy_helper
from onnx import shape_inference
import onnxruntime as ort
import numpy as np
import time

'''
test_data_dir = 'test_data_0'
input_tensor = np.asarray(cv2.resize(cv2.imread('image1.jpg'),(224,224))).reshape((1, 3, 224, 224)).astype(np.float32)
sess = ort.InferenceSession('conv.onnx')
output_tensor = sess.run(None, {'input:0': input_tensor})[0]
tensor_I = numpy_helper.from_array(input_tensor)
with open(os.path.join(test_data_dir, 'input_0.pb'), 'wb') as f:
    f.write(tensor_I.SerializeToString())
tensor_O = numpy_helper.from_array(output_tensor)
with open(os.path.join(test_data_dir, 'output_0.pb'), 'wb') as f:
    f.write(tensor_O.SerializeToString())
'''

np.random.seed(int(time.time()))
x = np.random.rand(8,3,1024,1024).astype(np.float32)
w = np.random.rand(1,3,32,32).astype(np.float32)
sess = ort.InferenceSession('conv.onnx', providers=['CPUExecutionProvider'])
y = sess.run(None, {'conv_X': x, 'conv_W': w})[0]
print (y)
print('done')