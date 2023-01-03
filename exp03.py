import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics

class ConvNet:
    """
    Deep feed-forward neural network for identifying digits from the MNIST data set. Treats 
    problem as a classification task.
    """
    def __init__(self):
        """
        The constructor. Here is where we will initialize the architecture and structural
        components of the training algorithm (e.g., which loss function and optimization strategy
        to use)
        """
        self.architecture = self._initialize_architecture()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.architecture.parameters())

    @staticmethod
    def _initialize_architecture():
        """
        This private method instantiates the overarching architecture of the neural network.
        :return: The overarching architecture of the network.
        """
        # below are the network layers we will be using to train and make prediction
        architecture = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),  # pooling layer
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),  #pooling layer
            nn.Flatten(),
            nn.Linear(256, 64),  # 256 in features
            nn.ReLU(),
            nn.Linear(64, 10)  # out size 10
        )
        return architecture

    def fit(self, x, y, epochs=25, batch_size=None, verbose=True):
        """
        Train the model to predict y when given x.
        :param x: The input/features data
        :param y: The the output/target data
        :param epochs: The number of times to iterate over the training data.
        :param batch_size: How many samples to consider at a time before updating network weights.
        sqrt(# of training instances) by default.
        :param verbose: Print a lot of info to the console about training.
        :return: None. Internally the weights of the network are updated.
        """
        if batch_size is None:
            batch_size = int(np.sqrt(len(x)))

        # Convert from Numpy to PyTorch tensor.
        x = torch.from_numpy(x).float()
        x = torch.reshape(x, (x.shape[0], 1, 28, 28))
        y = torch.from_numpy(y)

        # Train for the given number of epochs.
        for epoch in range(epochs):
            if verbose:
                print("Working on Epoch", epoch, "of", epochs)

            # Let the model know we're starting another round of training
            self.architecture.train()

            # Organize into helper methods which automatically put things into
            # well formed batches for us.
            train_ds = TensorDataset(x, y)
            train_dl = DataLoader(train_ds, batch_size=batch_size)

            # Iterate over each minibatch
            for x_minibatch, y_minibatch in train_dl:
                # Feed through the architecture
                pred_minibatch = self.architecture(x_minibatch)

                # Calculate the loss
                loss = self.loss_function(pred_minibatch, y_minibatch)

                # Apply backpropegation using the optimizer
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if verbose:
                print("Loss at end of epoch:", loss)


    def predict(self, x):
        """
        Predict an outcome given a sample of features.
        :param x: The features to use for a prediction.
        :return: The predicted outcome based on the features.
        """
        # Convert from Numpy to PyTorch tensor.
        x = torch.from_numpy(x).float()
        x = torch.reshape(x, (x.shape[0], 1, 28, 28)) # reshape data
        # Let the model know we're in "evaluate" mode (as opposed to training)
        self.architecture.eval()
        # Feed through the architecture.
        result = self.architecture(x)
        # Convert back to numpy
        result = result.detach().numpy()
        # Get max result (our prediction)
        result = np.argmax(result, axis=1)
        return result


class Exp03:

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        """

        train = np.loadtxt(file_path_prefix + "mnist_train.csv", delimiter=',', dtype=np.int64)
        test = np.loadtxt(file_path_prefix + "mnist_test.csv", delimiter=',', dtype=np.int64)

        x_train = train[:, 1:]
        y_train = train[:, 0].ravel()

        x_test = test[:, 1:]
        y_test = test[:, 0].ravel()

        return x_train, y_train, x_test, y_test

    def run(self):
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp03.load_train_test_data()

        print("Training Model...")

        # (1) Initialize model;
        model = ConvNet()

        # (2) Train model using the function 'fit' and the variables 'x_train'
        # and 'y_train'

        model.fit(x_train, y_train)

        print("Training complete!")
        print()

        print("Evaluating Model")
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        print("-- SciKit Learn Classification Report: Training Data")
        print(metrics.classification_report(y_train, y_train_pred))

        print("-- SciKit Learn Classification Report: Testing Data")
        print(metrics.classification_report(y_test, y_test_pred))

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)