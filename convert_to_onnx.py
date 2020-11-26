import sys
from scipy.special import softmax

import torch.onnx
import onnxruntime as ort
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from pytorch2keras.converter import pytorch_to_keras

from models.faceboxes import FaceBoxes

input_dim = 1024
num_classes = 2
model_path = "weights/FaceBoxesProd.pth"
net = FaceBoxes('train', input_dim, num_classes)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

net = load_model(net, model_path, False) 
net.eval()
net.to("cuda")

model_name = model_path.split("/")[-1].split(".")[0]
onnx_model_path = f"models/onnx/base-model.onnx"


# export ONNX model
dummy_input = torch.randn(1, 3, input_dim, input_dim).to("cuda")
torch.onnx.export(net, dummy_input, onnx_model_path, verbose=False, input_names=['input'], output_names=['output'])

"""
# try using pytorch2keras
keras_model = pytorch_to_keras(net, dummy_input, [(3, input_dim, input_dim)])
keras_model_path = f"models/onnx/base-model"
#keras_model.save(model_path)



# 0. print PyTorch outputs
out = net(dummy_input)
dummy_input = dummy_input.cpu().detach().numpy()
out = out.cpu().detach().numpy()
loc = out[:, :, 2:]
conf = out[:, :, :2]
scores = softmax(conf, axis=-1)
print(scores)


# 1. check if ONNX outputs are the same
ort_session = ort.InferenceSession(onnx_model_path)
input_name = ort_session.get_inputs()[0].name
out = ort_session.run(None, {input_name: dummy_input})[0]
loc = out[:, :, 2:]
conf = out[:, :, :2]
scores = softmax(conf, axis=-1)
print(scores)


# 2. check if Keras outputs are the same
keras_model_path = f"models/onnx/base-model"
keras_model = tf.keras.models.load_model(keras_model_path)
out = keras_model.predict(dummy_input)
loc = out[:, :, 2:]
conf = out[:, :, :2]
scores = softmax(conf, axis=-1)
print(scores)


# 3. check if intermediate results of Keras are the same
test_fn = K.function([keras_model.input], [keras_model.get_layer('334').output[0]])
test_out = test_fn(dummy_input)
print(np.round(np.array(test_out), 4)[:30])
"""

