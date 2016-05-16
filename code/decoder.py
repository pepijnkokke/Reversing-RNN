import theano
import theano.tensor as T
import numpy         as np
from gru import GRU

class Decoder(GRU):
    def __init__(self, E, enc_output, K, embedding_size=500, hidden_layer=1000):
        GRU.__init__(self, K, embedding_size, hidden_layer)

        self.E = E

        # additional weights for the initial input
        self.C = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        self.Cz = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        self.Cr = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        # the vocabulary
        self.vocab = theano.shared(np.eye(K))

        # the first word
        self.y0 = theano.shared(np.zeros(K))

        self.Gl = theano.shared(np.random.uniform(
            size=(K, embedding_size),
            low=-0.1, high=0.1))

        self.Gr = theano.shared(np.random.uniform(
            size=(embedding_size, hidden_layer),
            low=-0.1, high=0.1))

        self.Oh = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        self.Oy = theano.shared(np.random.uniform(
            size=(hidden_layer, K),
            low=-0.1, high=0.1))

        self.Oc = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

        # create the input and output variables of the decoder
        self.input  = enc_output
        self.output = self.dec_sentence()
        self.decode = theano.function(inputs=[self.input], outputs=self.output)

    def dec_word(self, y_tm1, h_tm1, c):
        """
        Input:
        y_tm1: the previously generated word (a K-dimensional vector)
        h_tm1: the state of the hidden layer before the current step
        c    : the output of the encoder

        Output:
        ht: the state of the hidden layer after the current step
        """

        # update and reset gate
        z = T.nnet.sigmoid(self.Wz.dot(self.E.dot(y_tm1)) + self.Uz.dot(h_tm1) + self.Cz.dot(c))
        r = T.nnet.sigmoid(self.Wr.dot(self.E.dot(y_tm1)) + self.Ur.dot(h_tm1) + self.Cr.dot(c))

        # candidate update
        h_candidate = T.tanh(self.W.dot(self.E.dot(y_tm1)) + self.U.dot(r * h_tm1) + self.C.dot(c))

        return z * (h_tm1) + (1 - z) * (h_candidate)

    def generate_word(self, y_tm1, c):
        h = self.dec_word(y_tm1, self.h, c)
        G = self.Gl.dot(self.Gr)
        s = self.Oh.dot(h) + self.Oy.dot(y_tm1) + self.Oc.dot(c)
        values = G.dot(s)

        word_idx = T.argmax(T.nnet.softmax(values), axis=1)
        return self.vocab[word_idx][0]

    def dec_sentence(self):
        result, _ = theano.scan(fn=self.generate_word,
                                outputs_info=[self.y0],
                                non_sequences=self.input,
                                n_steps=5)

        return result

