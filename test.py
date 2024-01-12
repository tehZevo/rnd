import os
import random
#suppress tf warnings for calling train_on_batch etc in quick succession
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

from rnd import RND

INPUT_SIZE = 32

rnd = RND(INPUT_SIZE)

rnd.predictor_model.summary()
rnd.target_model.summary()

samples = np.random.normal(0, 1, size=[10, INPUT_SIZE])

#rnd reward should decrease
for step in range(100):
  sample = random.choice(samples)
  rnd_reward = rnd.step(sample)
  print(f"Step {step + 1} reward: {rnd_reward}")

rnd.save("models")

rnd = RND()

rnd.load("models")

#rnd reward should already be low
loss = rnd.get_rnd_loss(samples)
print(f"Post-load loss: {loss}")
