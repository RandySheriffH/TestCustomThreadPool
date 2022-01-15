import numpy as np
from onnx import *
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime import SessionOptions

add_X = helper.make_tensor_value_info('add_X', TensorProto.INT32, [1024])
add_Y = helper.make_tensor_value_info('add_Y', TensorProto.INT32, [1024])
add_Z = helper.make_tensor_value_info('add_Z', TensorProto.INT32, [1024])
add = helper.make_node('Add', ['add_X', 'add_Y'], ['add_Z'])
add_graph = helper.make_graph([add], 'add_graph', [add_X, add_Y], [add_Z])
add_model = helper.make_model(add_graph, producer_name='add_model', opset_imports=[helper.make_opsetid("", 14)])
#add_model = helper.make_model(add_graph, producer_name='add_model')
onnx.save(add_model, 'add.onnx')

conv_X = helper.make_tensor_value_info('conv_X', TensorProto.FLOAT, [-1,3,1024,1024]) # NCHW
conv_W = helper.make_tensor_value_info('conv_W', TensorProto.FLOAT, [1,3,32,32])
conv_Y = helper.make_tensor_value_info('conv_Y', TensorProto.FLOAT, [-1,-1,-1,-1])
conv = helper.make_node('Conv', ['conv_X', 'conv_W'], ['conv_Y'])
conv_graph = helper.make_graph([conv], 'conv_graph', [conv_X, conv_W], [conv_Y])
conv_model = helper.make_model(conv_graph, producer_name='conv_model', opset_imports=[helper.make_opsetid("", 15)])
# conv_model = helper.make_model(conv_graph, producer_name='conv_model')
onnx.save(conv_model, 'conv.onnx')

print ('done')