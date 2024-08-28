"""Tests for model_utils.py."""

import numpy as np
from klearn_tcyclone.models_utils import predict_context_shift
from klearn_tcyclone.testing_utils import provide_TensorContextData, provide_model

def test_predict_context_shift():
    model = provide_model()
    tensor_context = provide_TensorContextData()

    stop = 100
    model.fit(tensor_context[:stop])

    initial_context = tensor_context[0]
    current_context = predict_context_shift(model, initial_context)
    assert np.all(current_context.data[:,:-1] == initial_context.data[:,1:])