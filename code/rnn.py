import sys
import theano
import theano.tensor as T
import numpy         as np
import read_data

class GRU:
    def __init__(self, K, hidden_layer=8):
        """
        K: dimensionality of the word embeddings
        hidden_layer: size of hidden layer
        """

        # state of the hidden layer
        self.h = theano.shared(np.zeros(hidden_layer))

        # input weights to the hidden layer and the gates
        self.W = theano.shared(np.random.uniform(
            size=(hidden_layer, K),
            low=-0.1, high=0.1))

        self.Wz = theano.shared(np.random.uniform(
            size=(hidden_layer, K),
            low=-0.1, high=0.1))

        self.Wr = theano.shared(np.random.uniform(
            size=(hidden_layer, K),
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


class Decoder(GRU):
    def __init__(self, K, hidden_layer=8):
        GRU.__init__(self, K, hidden_layer)

        # create the input and output variables of the decoder
        self.input  = T.vector()
        self.output = T.matrix()
        #self.decode = wisten we het maar

    def dec_word(self):
        pass




class Encoder(GRU):
    def __init__(self, K, hidden_layer=8):
        GRU.__init__(self, K, hidden_layer)

        # create the input and output variables of the encoder
        self.input  = T.matrix()
        self.output = self.enc_sentence(self.input)
        self.encode = theano.function(inputs=[self.input], outputs=[self.output],
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
        z = T.nnet.sigmoid(self.Wz.dot(x_t) + self.Uz.dot(h_tm1))
        r = T.nnet.sigmoid(self.Wr.dot(x_t) + self.Ur.dot(h_tm1))

        # candidate update
        h_candidate = T.tanh(self.W.dot(x_t) + self.U.dot(r * h_tm1))

        return (1 - z) * (h_tm1) + z * (h_candidate)


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


def encode_sentence(sentence, encoding, reversed=False):
    step = -1 if reversed else 1

    return np.array([encoding[word] for word in sentence.split(' ')[::step]])


def main():
    """
    Expects a path to the training data as the first argument, for example:
    python2 code/rnn.py data/en/qa1_single-supporting-fact_train.txt
    """
    sentences = read_data.get_sentences(sys.argv[1])
    encoding = read_data.sentences_to_word_encodings(sentences)

    # create the input (X) and desired output (Y)
    X = [encode_sentence(s, encoding) for s in sentences]
    Y = [encode_sentence(s, encoding, reversed=True) for s in sentences]

    encoder = Encoder(39)
    out = encoder.output
    error = out.dot(out)

    # just checking if the gradient can be computed
    grad = T.grad(error, wrt=encoder.Wr)
    grad_f = theano.function(inputs=[encoder.input], outputs=[grad])

    for sentence in X:
        encoder.encode(sentence)
        print encoder.h.eval()
        print grad_f(sentence)


if __name__ == '__main__':
    main()
