import numpy as np
from onnx import *
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime import SessionOptions

X = helper.make_tensor_value_info('X', TensorProto.INT32, [1024])
Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [1024])
Z = helper.make_tensor_value_info('Z', TensorProto.INT32, [1024])
add = helper.make_node("Add", ["X", "Y"], ["Z"])
graph = helper.make_graph([add], "graph", [X, Y], [Z])
model = helper.make_model(graph, producer_name='model')
onnx.save(model, 'model.onnx')
print ('done')