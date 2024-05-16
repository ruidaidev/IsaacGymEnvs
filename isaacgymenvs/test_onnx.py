import onnx
import os
import onnxruntime
import numpy as np
import torch

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

# load onnx model
model_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../policy")
model_path = os.path.join(model_root, 'CentauroCabinet.onnx')
model = onnx.load(model_path)

# check model structure
onnx.checker.check_model(model)

# print model information
print(onnx.helper.printable_graph(model.graph))

# create inference session
session = onnxruntime.InferenceSession(model_path)

# obtain input output information
input_name = session.get_inputs()[0].name
output_name_mu = session.get_outputs()[0].name
output_name_logstd = session.get_outputs()[1].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

# create a dummy input
# dummy_input = torch.randn(1, 47)
# dummy_input = np.random.rand(1, 47).astype(np.float32)
dummy_input = np.array([[5.6661e-02, -2.4466e-01, -1.0000e+00, -5.7106e-01, -4.3399e-01,
        -9.2090e-03,  5.7053e-01, -8.4939e-01,  1.1585e-01, -8.4476e-01,
        -2.3984e-02, -4.5031e-01,  5.8540e-01,  7.6763e-01, -1.3542e-01,
        -8.5595e-01,  3.2024e-02, -4.5329e-01, -1.0000e+00, -6.6963e-02,
        -1.1198e-01,  1.6188e-03,  1.7972e-03, -1.6402e-02,  9.9353e-03,
        -7.4485e-03,  5.6929e-02, -1.7647e-02, -1.1678e-03, -4.7604e-03,
         2.6297e-03, -1.0301e-03,  7.0560e-04,  9.8200e-02,  4.5989e-02,
        -3.3964e-02, -2.8623e-03,  5.3124e-02, -2.1677e-02, -2.0823e-05,
         7.9680e-04, -4.3040e-06, -4.9068e-01, -2.1793e-01, -1.8082e-01,
         0.0000e+00,  8.0645e-09]], dtype=np.float32)
# print(dummy_input)

# run session
mu = session.run([output_name_mu], {input_name: dummy_input})[0][0]
# logstd = session.run([output_name_logstd], {input_name: dummy_input})[0][0]
# sigma = np.exp(logstd)
# action_pre = np.random.normal(loc=mu, scale=sigma)
current_action = mu
num_actions = len(current_action)
low = np.ones(num_actions) * -1.
low[6:12] = np.zeros(6)
low[19:21] = np.zeros(2)
high = np.ones(num_actions) * 1.
high[6:12] = np.zeros(6)
high[19:21] = np.zeros(2)
rescaled_action = rescale_actions(low, high, np.clip(current_action, -1.0, 1.0))

print(rescaled_action)
# print(logstd)
# print(sigma)
# print(action_pre)