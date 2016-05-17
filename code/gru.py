import theano
import theano.tensor as T
import numpy         as np

class GRU:
    """
    Base class containing the GRU weights used by both the encoder and decoder
    """
    def __init__(self,
                 K,
                 embedding_size,
                 hidden_layer=8,
                 use_context_vector=False,
                 E=None):
        """
        K                  : dimensionality of the word embeddings
        embedding_size     : dimensionality of the word embeddings
        hidden_layer       : size of hidden layer
        use_context_vector : whether or not to use a context vector
        E                  : a word embedding to use (optional)
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

        # word embedding matrix
        if E is None:
            self.E = theano.shared(np.random.uniform(
                size=(embedding_size, K),
                low=-0.1, high=0.1), name='E')
        else:
            self.E = E

        self.params = [self.W, self.Wz, self.Wr,
                       self.U, self.Uz, self.Ur,
                       self.V, self.E]

        # additional weights for the context vector
        if use_context_vector:

            self.C = theano.shared(np.random.uniform(
                size=(hidden_layer, hidden_layer),
                low=-0.1, high=0.1), name='C')

            self.Cz = theano.shared(np.random.uniform(
                size=(hidden_layer, hidden_layer),
                low=-0.1, high=0.1), name='Cz')

            self.Cr = theano.shared(np.random.uniform(
                size=(hidden_layer, hidden_layer),
                low=-0.1, high=0.1), name='Cr')

            self.params.extend([self.C, self.Cz, self.Cr])


    def compute(self, x_t, h_tm1, c=None):
        """
        Input
        x_t    : the current word (a K-dimensional vector)
        h_tm1  : the state of the hidden layer before the current step

        Output
        h_t    : the state of the hidden layer after the current step
        """

        if c is None:

            z = T.nnet.sigmoid(self.Wz.dot(self.E.dot(x_t))
                               + self.Uz.dot(h_tm1))
            r = T.nnet.sigmoid(self.Wr.dot(self.E.dot(x_t))
                               + self.Ur.dot(h_tm1))
            h_cand = T.tanh(self.W.dot(self.E.dot(x_t))
                            + self.U.dot(r * h_tm1))

        else:

            z = T.nnet.sigmoid(self.Wz.dot(self.E.dot(x_t))
                               + self.Uz.dot(h_tm1)
                               + self.Cz.dot(c))
            r = T.nnet.sigmoid(self.Wr.dot(self.E.dot(x_t))
                               + self.Ur.dot(h_tm1)
                               + self.Cr.dot(c))
            h_cand = T.tanh(self.W.dot(self.E.dot(x_t))
                            + self.U.dot(r * h_tm1)
                            + self.C.dot(c))

        return (z * h_tm1) + ((1 - z) * h_cand)
