import keras
import keras.layers as layers
import tensorflow as tf
import os
import numpy as np

import pydot
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot

# draw_path = os.path.abspath(os.path.dirname(__file__)) + "\\structure\\several_demo_model.png"
# keras.utils.plot_model(self._model, draw_path, show_shapes=True)
