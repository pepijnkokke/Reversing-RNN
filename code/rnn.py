import sys
import theano
import theano.tensor as T
import numpy         as np
import read_data

from gru     import GRU
from encoder import Encoder
from decoder import Decoder


class Seq2Seq_RNN:
    def __init__(self, K):
        encoder_nnet = GRU(K, embedding_size=500, hidden_layer=1000)
        self.encoder = Encoder(encoder_nnet, K)
        decoder_nnet = GRU(K, embedding_size=500, hidden_layer=1000,
                           use_context_vector=True, E=encoder_nnet.E)
        self.decoder = Decoder(decoder_nnet, self.encoder.output,
                               K=K, embedding_size=500, hidden_layer=1000)
        self.run = theano.function(
            inputs  = [self.encoder.input, self.decoder.length],
            outputs = self.decoder.output,
            updates = [(encoder_nnet.h, self.encoder.output)
            ])

        self.run_final = theano.function(
            inputs  = [self.encoder.input, self.decoder.length],
            outputs = self.decoder.output_final,
            updates = [(encoder_nnet.h, self.encoder.output)])

        self.params = []
        self.params.extend(self.encoder.params)
        self.params.extend(self.decoder.params)

        # this kinda messes with the order of the parameters
        self.params = list(set(self.params))


def encode_sentence(sentence, encoding, reversed=False):
    step = -1 if reversed else 1

    return np.array([encoding[word] for word in sentence.split(' ')[::step]])


def decode_sentence(sentence, encoding):
    out = []
    for vec in sentence:
        v = vec.tolist()
        for word, enc in encoding.items():
            if enc == v:
                out.append(word)
                break

    return out


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

    rnn = Seq2Seq_RNN(K=39)

    ref_output = T.matrix()
    error = T.sum(abs(rnn.decoder.output - ref_output))
    error_f = theano.function(inputs=[rnn.decoder.output, ref_output],
                              outputs=error)

    gradient = T.grad(error, wrt=rnn.params)
    train_updates = [(p, p - 0.01 * grad_p)
                     for (p, grad_p) in zip(rnn.params, gradient)]

    train = theano.function(
        inputs  = [rnn.encoder.input, rnn.decoder.length, ref_output],
        outputs = error,
        updates = train_updates)

    n_epochs = int(sys.argv[2])
    train_reference_pairs = zip(X, Y)
    for i in range(n_epochs):
        print 'Epoch {}:'.format(i)
        train_idxs = np.random.choice(len(train_reference_pairs), 100)
        for idx in train_idxs:
            sentence, ref = train_reference_pairs[idx]
            print 'Error: {}'.format(train(sentence, len(sentence), ref))

    for sentence, ref in train_reference_pairs[:10]:
        s = rnn.run_final(sentence, len(sentence))
        print 'Sentence: {}'.format(' '.join(decode_sentence(sentence, encoding)))
        print 'Reversed: {}'.format(' '.join(decode_sentence(s, encoding)))
        print ''

if __name__ == '__main__':
    main()
