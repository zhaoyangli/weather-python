import numpy as np
from torch import float64

import sklearn.linear_model
import pandas as pd
import tensorflow as tf
############################################


import torch

# load and split data
dataset = pd.read_csv("../data/augmented_data.csv").to_numpy()
y = dataset[:, 1:3]
X = dataset[:, 3:]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
print("Split dataset to size:", len(X_train), len(X_test))
print(X_train.shape)
print(X_train.size)

# Dataset Manipulation
x_train = X_train.reshape(-1, 56, 1)
y_train = y_train[:, 1].reshape(-1, 1)
print(x_train.shape)
X = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)

x_test = X_test.reshape(-1, 56, 1)
y_test = y_test[:, 1].reshape(-1, 1)
X1 = torch.from_numpy(x_test)
y1 = torch.from_numpy(y_test)


class VanillaRNN(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=1, num_layers=2):
        super(VanillaRNN, self).__init__()
        """The initialization method is used to initialize all the components needed
        to implement our neural model. In this case, we have to initialize a 
        recurrnet neural network and a linear transformation."""

        # Initialize the RNN (using torch.nn.RNN). Use batch_first=True as input argument.
        self.rnn = torch.nn.RNN(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
                                , dtype=torch.double
                                )

        # Initialize the Linear transformation (using torch.nn.Linear)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=1
                                      , dtype=torch.double
                                      )

    def forward(self, X):
        """This function processes the input sequence X with an RNN. Then it applies
        a linear transformation on top of the last hidden state. In eval modality,
        a sigmoid followed by a threshold is applied to output the final prediction.

        Arguments
        ---------
        X : torch.Tensor
          Tensor containing the sequences (N, L, 1).

        Returns
        ---------
        out: torch.Tensor
          Tensor containing the logits in training modality or the predictions
          in eval modality (N,1).
        """

        # Run the RNN
        Z, _ = self.rnn(X)

        # Making the hidden states visible from outside (needed for 1.3)
        self.Z = Z

        # Select the last hidden state
        z = Z[:, -1, :]

        # Apply linear transformation
        out = self.linear(z)

        # Managing eval mode
        if not self.training:
            # Remember to apply a threshold on top of the sigmoid to get the prediction.
            out = torch.sigmoid(out) > 0.5

        return out


def data_generation(L, N, D=1, prob=0.05):
    """This function generates random sequences of 0 and 1.
    The label is 1 if there is at least one element of the
    sequence that contains 1.

    Arguments
    ---------
    L : int
      Lenght of the sequences.
    N: int
      Number of examples.
    D: int
      Number of output features.
    prob: float:
      Probability to draw a 1.

    Returns
    ---------
    X: torch.Tensor
      Tensor of dimensionality (L,N,D) containing
      the generated sequences.
    y: torch.Tensor
      Tensor of dimensionality (N,1) containing the labels.
    """
    X = torch.bernoulli(torch.full((N, L, D), prob)).float()
    y = torch.any(X >= 1.0, dim=1).float()
    return X, y


rnn = VanillaRNN(input_size=1, hidden_size=4, num_layers=2)
rnn.train()
out = rnn(X[0:100, :, :])
assert out.shape == (100, 1), "The output of the RNN has an unexpected shape"
rnn.eval()
out = rnn(X[0:100, :, :])
assert out.shape == (100, 1), "The output of the RNN has an unexpected shape (in eval mode)"
cnt_0 = torch.count_nonzero(out)
cnt_1 = torch.count_nonzero(1 - out.int())
assert cnt_0 + cnt_1 == 100, "The prediction returned in eval model must be only be 0. \
                              Make sure the threshold is applied"
print("Correct!")

lr = 0.01
batch_size = 100
num_epoch = 500
hidden_size = 1
num_layers = 3

# Initialize the Vanilla RNN
rnn = VanillaRNN(hidden_size=hidden_size, num_layers=num_layers)

# Initialize the Loss. Please, use torch.nn.BCEWithLogitsLoss
loss = torch.nn.BCEWithLogitsLoss()

# Initialize the Optimizer. Please, use torch.optim.Adam (with the lr specified above)
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

# Training Loop
for epoch in range(num_epoch):
    for i in range(0, 1353, batch_size):
        # Read minibatches (for both X and y)
        Xi = X[i:i + batch_size]
        yi = y[i:i + batch_size]

        # Run the model
        logits = rnn(X)

        # Compute the loss (use l as variable for the loss)
        l = loss(logits, y)

        # Update the parameters
        rnn.zero_grad()
        l.backward()
        optimizer.step()

        # Print loss
    if (epoch + 1) % 50 == 0:
        print("Epoch %03d: Train_loss: %.4f " % (epoch + 1, l.item()))

# Switch model to eval mode
rnn.eval()

# Run prediction
pred = rnn(X1)
pred = pred * 1
# Print prectiction
# print(pred)
mse = tf.keras.losses.MeanSquaredError()
mse(pred, y1).numpy()
ac = tf.keras.metrics.Accuracy()
ac.update_state(pred, y1)
print(mse(pred, y1).numpy())
print(ac.result().numpy())

###############################################
# batch_size = 100
# num_epoch = 500
# hidden_size = 5
# num_layers = 2
# Epoch 050: Train_loss: 0.6434
# Epoch 100: Train_loss: 0.6310
# Epoch 150: Train_loss: 0.6734
# Epoch 200: Train_loss: 0.6386
# Epoch 250: Train_loss: 0.6430
# Epoch 300: Train_loss: 0.6505
# Epoch 350: Train_loss: 0.6511
# Epoch 400: Train_loss: 0.6311
# Epoch 450: Train_loss: 0.6248
# Epoch 500: Train_loss: 0.6263
##################################################################
# lr = 0.01
# batch_size = 100
# num_epoch = 500
# hidden_size = 3
# num_layers = 2
# Epoch 050: Train_loss: 0.6599
# Epoch 100: Train_loss: 0.6731
# Epoch 150: Train_loss: 0.6684
# Epoch 200: Train_loss: 0.6662
# Epoch 250: Train_loss: 0.6637
# Epoch 300: Train_loss: 0.6588
# Epoch 350: Train_loss: 0.6542
# Epoch 400: Train_loss: 0.6669
# Epoch 450: Train_loss: 0.6827
# Epoch 500: Train_loss: 0.6826
# 0.504424778761062
##################################################################
# lr = 0.01
# batch_size = 100
# num_epoch = 500
# hidden_size = 1
# num_layers = 2
# Epoch 050: Train_loss: 0.6797
# Epoch 100: Train_loss: 0.6800
# Epoch 150: Train_loss: 0.6823
# Epoch 200: Train_loss: 0.6823
# Epoch 250: Train_loss: 0.6822
# Epoch 300: Train_loss: 0.6822
# Epoch 350: Train_loss: 0.6838
# Epoch 400: Train_loss: 0.6837
# Epoch 450: Train_loss: 0.6836
# Epoch 500: Train_loss: 0.6836
# 0.5088495575221239
##################################################################
# lr = 0.02
# batch_size = 100
# num_epoch = 500
# hidden_size = 2
# num_layers = 1
# Epoch 050: Train_loss: 0.6800
# Epoch 100: Train_loss: 0.6797
# Epoch 150: Train_loss: 0.6797
# Epoch 200: Train_loss: 0.6799
# Epoch 250: Train_loss: 0.6800
# Epoch 300: Train_loss: 0.6800
# Epoch 350: Train_loss: 0.6800
# Epoch 400: Train_loss: 0.6800
# Epoch 450: Train_loss: 0.6803
# Epoch 500: Train_loss: 0.6801
# 0.47123893805309736

##################################################################
# lr = 0.01
# batch_size = 100
# num_epoch = 500
# hidden_size = 4
# num_layers = 3
# Epoch 050: Train_loss: 0.6669
# Epoch 100: Train_loss: 0.6649
# Epoch 150: Train_loss: 0.6661
# Epoch 200: Train_loss: 0.6521
# Epoch 250: Train_loss: 0.6609
# Epoch 300: Train_loss: 0.6706
# Epoch 350: Train_loss: 0.6531
# Epoch 400: Train_loss: 0.6499
# Epoch 450: Train_loss: 0.6426
# Epoch 500: Train_loss: 0.6454
# 0.48451327433628316

##################################################################
# lr = 0.01
# batch_size = 50
# num_epoch = 500
# hidden_size = 1
# num_layers = 4
# Epoch 050: Train_loss: 0.6809
# Epoch 100: Train_loss: 0.6859
# Epoch 150: Train_loss: 0.6812
# Epoch 200: Train_loss: 0.6808
# Epoch 250: Train_loss: 0.6901
# Epoch 300: Train_loss: 0.6901
# Epoch 350: Train_loss: 0.6901
# Epoch 400: Train_loss: 0.6901
# Epoch 450: Train_loss: 0.6901
# Epoch 500: Train_loss: 0.6901
# 0.4778761061946903

##################################################################


# lr = 0.01
# # batch_size = 100
# # num_epoch = 500
# # hidden_size = 1
# # num_layers = 2
# Epoch 050: Train_loss: 0.6778
# Epoch 100: Train_loss: 0.6750
# Epoch 150: Train_loss: 0.6750
# Epoch 200: Train_loss: 0.6750
# Epoch 250: Train_loss: 0.6750
# Epoch 300: Train_loss: 0.6750
# Epoch 350: Train_loss: 0.6757
# Epoch 400: Train_loss: 0.6808
# Epoch 450: Train_loss: 0.6768
# Epoch 500: Train_loss: 0.6899
# 0.4778761061946903
# 0.5221239


##################################################################
# lr = 0.01
# batch_size = 50
# num_epoch = 500
# hidden_size = 1
# num_layers = 3
# Epoch 050: Train_loss: 0.6817
# Epoch 100: Train_loss: 0.6854
# Epoch 150: Train_loss: 0.6822
# Epoch 200: Train_loss: 0.6856
# Epoch 250: Train_loss: 0.6876
# Epoch 300: Train_loss: 0.6875
# Epoch 350: Train_loss: 0.6873
# Epoch 400: Train_loss: 0.6863
# Epoch 450: Train_loss: 0.6863
# Epoch 500: Train_loss: 0.6863
# 0.45132743362831856

##################################################################
