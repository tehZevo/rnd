from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def create_model(input_size, output_size=32, hidden_sizes=[32],
  activation="swish"
):
  model = Sequential()
  #TODO: add batch_norm?
  model.add(InputLayer([input_size]))
  for size in hidden_sizes:
    model.add(Dense(size, activation=activation))
  model.add(Dense(output_size))

  return model

def create_models(input_size, output_size=32, predictor_hidden=[32],
  target_hidden=[32, 32], activation="swish", lr=1e-3
):
  predictor = create_model(input_size, output_size, predictor_hidden, activation)
  target = create_model(input_size, output_size, target_hidden, activation)

  predictor.compile(loss="mse", optimizer=Adam(lr))
  #compile target just so loading doesn't give us a warning
  target.compile(loss="mse", optimizer="sgd")

  return predictor, target
