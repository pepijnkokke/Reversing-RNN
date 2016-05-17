import theano
import theano.tensor as T
import numpy         as np


class Encoder:
    def __init__(self, nnet, K, embedding_size=500, hidden_layer=1000):
        """
        K              : dimensionality of the word embeddings
        embedding_size : dimensionality of the word embeddings
        hidden_layer   : size of hidden layer
        """
        #GRU.__init__(self, K=K, hidden_layer=hidden_layer,
        #             embedding_size=embedding_size)

        # create the input and output variables of the encoder
        self.nnet   = nnet
        self.params = nnet.params
        self.input  = T.matrix(name='input')
        self.output = self.enc_sentence(self.input)
        self.encode = theano.function(
            inputs  = [self.input],
            outputs = self.output,
            updates = [(self.nnet.h, self.output)])

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
        results, _ = theano.scan(
            lambda x_t, h_tm1: self.nnet.compute(x_t, h_tm1),
            outputs_info = self.nnet.h,
            sequences    = [input]
        )

        # we're only interested in the final state
        result = results[-1]

        return T.tanh(T.dot(self.nnet.V,result))
