import os

from tensorflow.keras.models import load_model
import numpy as np

from .utils import create_models

class RND:
  def __init__(self, input_size, output_size=32, predictor_hidden=[32],
    target_hidden=[32, 32], activation="swish", lr=1e-3, buffer_size=10000,
    batch_size=32
  ):
    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.buffer = []

    self.predictor_model, self.target_model = create_models(
      input_size=input_size,
      output_size=output_size,
      predictor_hidden=predictor_hidden,
      target_hidden=target_hidden,
      activation=activation,
      lr=lr
    )

  def save(self, path):
    os.makedirs(path, exist_ok=True)
    self.predictor_model.save(os.path.join(path, "predictor.keras"))
    self.target_model.save(os.path.join(path, "target.keras"))

  def load(self, path):
    self.predictor_model = load_model(os.path.join(path, "predictor.keras"))
    self.target_model = load_model(os.path.join(path, "target.keras"))

  def add_to_buffer(self, xs):
    if len(xs.shape) < 2:
      xs = np.expand_dims(xs, 0)
    for x in xs:
      self.buffer.append(x)

    while len(self.buffer) > self.buffer_size:
      self.buffer.pop(0)

  def get_batch(self, size):
    ii = np.random.randint(len(self.buffer), size=[size])
    x = np.array([self.buffer[i] for i in ii])
    y = self.target_model.predict_on_batch(x)
    return x, y

  def train(self):
    x, y = self.get_batch(self.batch_size)
    loss = self.predictor_model.train_on_batch(x, y)
    return loss

  def get_rnd_loss(self, x):
    if len(x.shape) < 2:
      x = np.expand_dims(x, 0)
    #NOTE: could OOM if x/y too large
    y = self.target_model.predict_on_batch(x)
    loss = self.predictor_model.test_on_batch(x, y)

    return loss

  def step(self, x):
    #calculate rnd loss
    rnd_loss = self.get_rnd_loss(x)
    #add sample to buffer
    self.add_to_buffer(x)
    #train for one step
    train_loss = self.train()

    return rnd_loss
