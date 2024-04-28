import numpy as np
import torch
from numpy import ndarray
from torch import nn, Tensor, as_tensor

BATCHES_BEFORE_PRINTING_LOSS = 10
DEFAULT_DTYPE = torch.float32

MIN_NUM_NEIGHBOURS = 5

class NormalsClusterClassifier(nn.Module):
    """
    Use softmax linear regression to train and predict the 3D-surface class of points in a point cloud.
    The model's features are based on the average intra-cluster distances of clustered surface normals.
    """
    def __init__(
            self,
            n_inputs: int,
            n_classes: int,
            max_iter: int,
            learning_rate: float,
            weight_decay: float = 0,
            init_scale: int = 1,
            batch_size: int = 1,
            device: str = None,
    ):
        super().__init__()
        self._n_inputs = n_inputs
        self._n_classes = n_classes
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._init_scale = init_scale
        self._batch_size = batch_size
        self._device = device
        self.build()

    def cast(self, np_array: ndarray, dtype=DEFAULT_DTYPE):
        """Make a tensor of `DEFAULT_DTYPE` from a numpy array stored onto `self._device`."""
        return as_tensor(np_array, dtype=dtype, device=self._device)

    def build(self):
        self._linear = nn.Linear(in_features=self._n_inputs, out_features=self._n_classes, device=self._device)
        self._loss_function = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

    def forward(self, X: Tensor):
        prediction = self._linear.forward(X)
        return prediction

    def fit(self, X: ndarray, y: ndarray, num_neighbours: ndarray):
        h, w, d = X.shape
        X = self.cast(X.reshape(h*w, d))
        y = self.cast(y.reshape(h*w), dtype=torch.int64)
        num_neighbours = self.cast(num_neighbours, dtype=torch.int64)
        enough_neighbour_indexes = num_neighbours > MIN_NUM_NEIGHBOURS
        X = X[enough_neighbour_indexes]
        y = y[enough_neighbour_indexes]
        for iter_num in range(self._max_iter):
            self._optimizer.zero_grad()
            batch_indexes = torch.as_tensor(
                np.random.choice(X.shape[0], size=self._batch_size, replace=False)
            )
            yhat = self(X[batch_indexes])
            loss = self._loss_function(yhat, y[batch_indexes])

            loss.backward()
            self._optimizer.step()

            if iter_num % BATCHES_BEFORE_PRINTING_LOSS == 0:
                print(f"Iteration {iter_num:>10,}: loss = {loss:>6.3f}")

    def predict(self, X: ndarray):
        with torch.no_grad():
            h, w, d = X.shape
            X = self.cast(X.reshape(h*w, d))
            Z = self(X)
            predictions = torch.argmax(Z, dim=1).reshape(h, w)
            return predictions.cpu().numpy()
