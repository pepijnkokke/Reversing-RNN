import theano
import theano.tensor as T
import numpy         as np
from gru import GRU

class Encoder(GRU):
    def __init__(self, K, embedding_size=500, hidden_layer=1000):
        """
        K: dimensionality of the word embeddings
        embedding_size: dimensionality of the word embeddings
        hidden_layer: size of hidden layer
        """
        GRU.__init__(self, K=K, hidden_layer=hidden_layer,
                     embedding_size=embedding_size)

        # word embedding matrix
        self.E = theano.shared(np.random.uniform(
            size=(embedding_size, K),
            low=-0.1, high=0.1), name='E')

        # create the input and output variables of the encoder
        self.input  = T.matrix(name='input')
        self.output = self.enc_sentence(self.input)
        self.encode = theano.function(inputs=[self.input], outputs=self.output,
                                      updates=[(self.h, self.output)])


    def enc_word(self, x_t, h_tm1):
        """
        Input:
        x_t: the current word (a K-dimensional vector)
        h_tm1: the state of the hidden layer before the current step

        Output:
        ht: the state of the hidden layer after the current step
        """

        # update and reset gate
        z = T.nnet.sigmoid(self.Wz.dot(self.E.dot(x_t)) + self.Uz.dot(h_tm1))
        r = T.nnet.sigmoid(self.Wr.dot(self.E.dot(x_t)) + self.Ur.dot(h_tm1))

        # candidate update
        h_candidate = T.tanh(self.W.dot(self.E.dot(x_t)) + self.U.dot(r * h_tm1))

        return z * (h_tm1) + (1 - z) * (h_candidate)


    def enc_sentence(self, input):
        """
        Input: a Theano matrix variable representing an input sentence
        Output: The variable holding the result
        """

        # This scan is basically a reduce operation.
        #
        # The given lambda function gets passed the elements of the sequences
        # given in `sequences`, followed by the state of the previous
        # iteration (whose initial value is given in `outputs_info`).
        #
        # It iterates over each row (first dimension) of the matrix `xs` and
        # returns a list giving each intermediate value, along with a list of
        # updates which we don't use.
        results, _ = theano.scan(lambda x_t, h_tm1: self.enc_word(x_t, h_tm1),
                                 outputs_info=self.h,
                                 sequences=[input])

        # we're only interested in the final state
        result = results[-1]

        return T.tanh(self.V.dot(result))
