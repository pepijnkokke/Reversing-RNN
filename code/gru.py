import theano
import theano.tensor as T
import numpy         as np

class GRU:
    """
    Base class containing the GRU weights used by both the encoder and decoder
    """
    def __init__(self, K, embedding_size, hidden_layer=8):
        """
        K: dimensionality of the word embeddings
        embedding_size: dimensionality of the word embeddings
        hidden_layer: size of hidden layer
        """

        # state of the hidden layer
        self.h = theano.shared(np.zeros(hidden_layer), name='h')

        # input weights to the hidden layer and the gates
        self.W = theano.shared(np.random.uniform(
            size=(hidden_layer, embedding_size),
            low=-0.1, high=0.1), name='W')

        self.Wz = theano.shared(np.random.uniform(
            size=(hidden_layer, embedding_size),
            low=-0.1, high=0.1), name='Wz')

        self.Wr = theano.shared(np.random.uniform(
            size=(hidden_layer, embedding_size),
            low=-0.1, high=0.1), name='Wr')

        # recurrent weights for the hidden layer and the gates
        self.U = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1), name='U')

        self.Uz = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1), name='Uz')

        self.Ur = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1), name='Ur')

        # the extra transformation between the encoder and decoder
        self.V = theano.shared(np.eye(hidden_layer))
