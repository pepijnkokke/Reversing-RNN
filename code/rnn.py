import sys
import theano
import theano.tensor as T
import numpy         as np
import read_data


class Encoder:
    def __init__(self, K, hidden_layer=8, e=lambda x: x):
        """
        K: dimensionality of the word embeddings
        hidden_layer: size of hidden layer
        e: word embedding function
        """

        self.e = e

        # state of the hidden layer
        self.h = theano.shared(np.zeros(hidden_layer))

        # input weights
        self.Whx = theano.shared(np.random.uniform(
            size=(hidden_layer, K),
            low=-0.1, high=0.1))

        # recurrent hidden layer weights
        self.Whh = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1))

    def feed_word(self, x_t, h_tm1):
        """
        Input:
        x_t: the current word (a K-dimensional vector)
        h_tm1: the state of the hidden layer before the current step

        Output:
        htj: the state of the hidden layer after the current step
        """
        e = self.e(x_t)
        ht = T.nnet.sigmoid(self.Whx.dot(e) + self.Whh.dot(h_tm1))

        return ht

    def feed_sentence(self):
        """
        Output:
        A function that updates the hidden state based on a given matrix of
        words.
        """

        # the variable used as input to the function
        input = T.matrix()

        # This scan is basically a reduce operation.
        #
        # The given lambda function gets passed the elements of the sequences
        # given in `sequences`, followed by the state of the previous
        # iteration (whose initial value is given in `outputs_info`).
        #
        # It iterates over each row (first dimension) of the matrix `xs` and
        # returns a list giving each intermediate value, along with a list of
        # updates which we don't use.
        results, _ = theano.scan(lambda x_t, h_tm1: self.feed_word(x_t, h_tm1),
                                 outputs_info=self.h,
                                 sequences=[input])

        # we're only interested in the final state
        result = results[-1]

        f = theano.function(inputs=[input], outputs=result,
                            updates=[(self.h, result)])

        return f


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
    f = encoder.feed_sentence()

    for sentence in X:
        f(sentence)
        print encoder.h.eval()


if __name__ == '__main__':
    main()
