# dnnpy Framework

## Installation
````bash
pip install dnnpy
````
## Example usage
```python
from dnnpy.activations import ReLU
from dnnpy.data import make_regression_data
from dnnpy.layers import Sequential, Dense, Dropout
from dnnpy.loss_functions import MAELoss
from dnnpy.optimizers import Adam
from dnnpy.train import train
from dnnpy.utils import split_data
import matplotlib.pyplot as plt

n_inputs = 10
hidden_units = 32
n_outputs = 1

x, y = make_regression_data(n_samples=1000, n_features=n_inputs, n_labels=1)

(x_train, y_train), (x_test, y_test) = split_data(x, y, ratio=0.7)

model = Sequential(Dense(in_features=n_inputs, out_features=hidden_units, activation=ReLU()),
                   Dropout(0.3),
                   Dense(in_features=hidden_units, out_features=n_outputs))

opt = Adam(model.parameters(), lr=1e-3)
loss_func = MAELoss()

train_loss, valid_loss = train(data=(x_train, y_train), network=model, loss=loss_func, optimiser=opt, epochs=30,
                               batch_size=16)

plt.plot(train_loss, label='train')
plt.plot(valid_loss, label='val')
plt.legend()
plt.show()
```