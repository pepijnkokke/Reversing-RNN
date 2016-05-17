import theano
import theano.tensor as T
import numpy         as np


class Decoder:
    def __init__(self, nnet, enc_output, K, embedding_size=500, hidden_layer=1000):
        """
        E              : the word embeddings used by the encoder
        enc_output     : the output variable of the encode
        K              : dimensionality of the word embeddings
        embedding_size : dimensionality of the word embeddings
        hidden_layer   : size of hidden layer
        """
        #GRU.__init__(self, K, embedding_size, hidden_layer, True, E)
        self.nnet   = nnet

        # length of the next desired output sentence for training
        self.length = T.scalar(dtype='int32')

        # the vocabulary (an identity matrix where each row is a one-hot vector
        # representing a word)
        self.vocab = theano.shared(np.eye(K), name='vocabulary')

        # the first word, which is just a vector of zeros
        self.y0 = theano.shared(np.zeros(K), name='y0')

        # additional variables used in the output
        self.Gl = theano.shared(np.random.uniform(
            size=(K, embedding_size),
            low=-0.1, high=0.1), name='Gl')

        self.Gr = theano.shared(np.random.uniform(
            size=(embedding_size, hidden_layer),
            low=-0.1, high=0.1), name='Gr')

        self.Oh = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1), name='Oh')

        self.Oy = theano.shared(np.random.uniform(
            size=(hidden_layer, K),
            low=-0.1, high=0.1), name='Oy')

        self.Oc = theano.shared(np.random.uniform(
            size=(hidden_layer, hidden_layer),
            low=-0.1, high=0.1), name='Oc')

        # update the references to include the new parameters
        self.params = nnet.params
        self.params.extend([self.Gl, self.Gr, self.Oc, self.Oh, self.Oy])

        # create the input and output variables of the decoder
        self.input  = T.dot(self.nnet.V,enc_output) # T.tanh(?)
        self.output = self.dec_sentence()
        self.decode = theano.function(
            inputs  = [self.input, self.length],
            outputs = self.output)

        self.output_final = self.dec_sentence_final()
        self.decode_final = theano.function(
            inputs  = [self.input, self.length],
            outputs = self.output_final)


    def generate_word(self, y_tm1, ht, c):
        h = self.nnet.compute(y_tm1, ht, c)
        G = self.Gl.dot(self.Gr)
        s = self.Oh.dot(h) + self.Oy.dot(y_tm1) + self.Oc.dot(c)
        values = G.dot(s)

        # returning the softmax distribution without argmaxing to select a word
        # makes the function differentiable
        word_idx = T.nnet.softmax(values)
        return word_idx[0], h


    def dec_sentence(self):
        """
        Decode a sentence
        """
        result, _ = theano.scan(
            fn            = self.generate_word,
            outputs_info  = [self.y0, self.nnet.h],
            non_sequences = self.input,
            n_steps       = self.length)

        return result[0]


    def generate_word_final(self, y_tm1, ht, c):
        h = self.nnet.compute(y_tm1, ht, c)
        G = self.Gl.dot(self.Gr)
        s = self.Oh.dot(h) + self.Oy.dot(y_tm1) + self.Oc.dot(c)
        values = G.dot(s)

        word_idx = T.argmax(T.nnet.softmax(values), axis=1)
        return self.vocab[word_idx][0], h


    def dec_sentence_final(self):
        """
        Decode a sentence
        """
        result, _ = theano.scan(
            fn            = self.generate_word_final,
            outputs_info  = [self.y0, self.nnet.h],
            non_sequences = self.input,
            n_steps       = self.length)

        return result[0]
