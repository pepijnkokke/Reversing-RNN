import theano
import theano.tensor as T
import numpy         as np

class GRU:
    def __init__(self, K, embedding_size, hidden_layer=8):
        """
        K: dimensionality of the word embeddings
        hidden_layer: size of hidden layer
        """

        # state of the hidden layer
        self.h = theano.shared(np.zeros(hidden_layer))

        # input weights to the hidden layer and the gates
        self.W = theano.shared(np.random.uniform(
            size=(hidden_layer, embedding_size),
            low=-0.1, high=0.1))

        self.Wz = theano.shared(np.random.uniform(
            size=(hidden_layer, embedding_size),
            low=-0.1, high=0.1))

        self.Wr = theano.shared(np.random.uniform(
            size=(hidden_layer, embedding_size),
            low=-0.1, high=0.1))

        # recurrent weights for the hidden layer and the gates
        self.U = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        self.Uz = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        self.Ur = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        # bonus extra final happy fun time transformation (linear)
        self.V = theano.shared(np.eye(hidden_layer))
