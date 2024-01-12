# Random Network Distillation
*Based loosely on https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938*

## Example usage
### Create an RND object
```python
from rnd import RND

#all parameters are optional
rnd = RND(
  input_size=128,
  output_size=32,
  predictor_hidden=[32], #hidden layer sizes of the predictor model, defaults to [32]
  target_hidden=[32, 32], #as above, but for target model, defaults to [32, 32]
  activation="swish", #defaults to swish
  lr=1e-3, #defailts to 1e-3
  buffer_size=10000, #training buffer size, defaults to 10000
  batch_size=32, #training batch size, defaults to 32
)
```

### Train RND and get loss in a single step
```python
import numpy as np

x = np.random.uniform(0, 1, size=[128]) #or however you obtain observation vectors
loss = rnd.step(x)
print(loss)
```

### Save/load RND models
```python
rnd.save("my/path/to/models")

#buffer and batch sizes can be adjusted, but other model creation parameters are ignored
new_rnd = RND(
  buffer_size=20000,
  batch_size=128
)
new_rnd.load("my/path/to/models")
```
