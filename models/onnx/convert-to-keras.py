import numpy as np
import cv2

import onnx
import onnxmltools
import onnx2keras
from onnx2keras import onnx_to_keras

# convert to keras
onnx_path = './base-model.onnx'
keras_path = onnx_path.split('/')[-1].split('.')[0]

onnx_model = onnxmltools.utils.load_model(onnx_path)
keras_model = onnx_to_keras(onnx_model, ['input'])
keras_model.save(keras_path)
keras_model.summary()
