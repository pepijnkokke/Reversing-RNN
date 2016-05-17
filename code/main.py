import cPickle
import theano
import theano.tensor as T
import numpy         as np
import sys
import read_data
import os


from rnn     import RNN
from gru     import GRU
from encoder import Encoder
from decoder import Decoder


EMBEDDING_SIZE = 500
DECODER_SIZE   = 1000
ENCODER_SIZE   = 1000
NN_CLASS_DICT  = { "RNN": RNN, "GRU": GRU }


class EncDec:
    def __init__(self, NNClass, K):

        encoder_nnet = NNClass(
            K, EMBEDDING_SIZE, ENCODER_SIZE,
            use_context_vector=False, E=None)
        self.encoder = Encoder(
            encoder_nnet,
            K, EMBEDDING_SIZE, ENCODER_SIZE)

        decoder_nnet = NNClass(
            K, EMBEDDING_SIZE, DECODER_SIZE,
            use_context_vector=True, E=encoder_nnet.E)
        self.decoder = Decoder(
            decoder_nnet, self.encoder.output,
            K, EMBEDDING_SIZE, DECODER_SIZE)

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


def encode_sentence(sentence, encoding):
    toks = read_data.sentence_to_tokens(sentence)
    return np.array([encoding[tok] for tok in toks])


def decode_sentence(sentence, encoding):
    out = []
    for vec in sentence:
        v = vec.tolist()
        for word, enc in encoding.items():
            if enc == v:
                out.append(word)
                break
    return out


def reverse_task(NNClass, n_epochs, sentences, K, encoding):
    """
    Attempt to train an RNN Encoder-Decoder to reverse sentences.
    """

    # create the input (X) and desired output (Y)
    X = [encode_sentence(s[1:], encoding) for s in sentences]
    Y = [encode_sentence(reversed(s[1:]), encoding) for s in sentences]

    encdec = EncDec(NNClass, K=K)

    ref_output = T.matrix()
    error   = T.sum(abs(encdec.decoder.output - ref_output))
    error_f = theano.function(
        inputs  = [encdec.decoder.output, ref_output],
        outputs = error)

    gradient      = T.grad(error, wrt=encdec.params)
    train_updates = [(p, p - 0.01 * grad_p)
                         for (p, grad_p) in zip(encdec.params, gradient)]

    train = theano.function(
        inputs  = [encdec.encoder.input, encdec.decoder.length, ref_output],
        outputs = error,
        updates = train_updates)

    train_reference_pairs = zip(X, Y)
    for i in range(n_epochs):
        print 'Epoch {}:'.format(i)
        train_idxs = np.random.choice(len(train_reference_pairs), 100)
        for idx in train_idxs:
            sentence, ref = train_reference_pairs[idx]
            print 'Error: {}'.format(train(sentence, len(ref), ref))

    for sentence, ref in train_reference_pairs[:10]:
        s = encdec.run_final(sentence, len(ref))
        print 'Sentence: {}'.format(' '.join(decode_sentence(sentence, encoding)))
        print 'Reversed: {}'.format(' '.join(decode_sentence(s, encoding)))
        print ''


def answer_task(NNClass, n_epochs, sentences, K, encoding):
    """
    Attempt to train an RNN Encoder-Decoder to do question answering.
    """

    encdec        = EncDec(NNClass, K=K)
    correct       = T.matrix('correct')
    error         = T.sum(abs(encdec.decoder.output - correct))
    error_f       = theano.function(
        inputs    = [encdec.decoder.output, correct],
        outputs   = error)
    error_grad    = T.grad(error, wrt=encdec.params)
    train_updates = [(p, p - 0.01 * grad_p)
                     for (p, grad_p) in zip(encdec.params, error_grad)]
    train_f       = theano.function(
        inputs    = [encdec.encoder.input, encdec.decoder.length, correct],
        outputs   = error,
        updates   = train_updates)

    for i in range(n_epochs):
        errs = []
        for j, s in enumerate(sentences):
            if '?' in s: # Answer required
                q, a = s.split('?')
                q_   = encode_sentence(q+'?', encoding)
                a_   = encode_sentence(a    , encoding)
                err  = train_f(q_, len(a_), a_)
                errs.append(err / float(len(a_)))
            else:
                s_ = encode_sentence(s, encoding)
                encdec.encoder.encode(s_)

            sys.stdout.write("\r{}: {}/{}".format(i, j,len(sentences)))
            sys.stdout.flush()

        print("\nAverage Error: {}".format(sum(errs) / float(len(errs))))

    for s in sentences:
        if '?' in s: # Answer required
            q, a = s.split('?')
            q_   = encode_sentence(q+'?', encoding)
            a_   = encode_sentence(a    , encoding)
            est  = encdec.run_final(q_,len(a_))
            print("Q: {}? A: {}? A: {}.".format(
                q, decode_sentence(est, encoding), a))
        else:
            s_ = encode_sentence(s, encoding)
            encdec.encoder.encode(s_)



def main():
    """
    Expects a path to the training data as the first argument, for example:
    python2 code/main.py GRU 10 data/en/qa1_single-supporting-fact_train.txt
    """
    if len(sys.argv) != 5:

        print "Usage: python2 main.py GRU 10 [DATA_SET] [reverse|answer]"

    else:

        NNClass   = NN_CLASS_DICT[sys.argv[1]]
        n_epochs  = int(sys.argv[2])
        data_file = sys.argv[3]
        TASK_DICT = { "reverse": reverse_task, "answer": answer_task }
        task_func = TASK_DICT[sys.argv[4]]

        outp_file = os.path.join(
            os.path.dirname(data_file),
            os.path.basename(data_file)+'.'+str(n_epochs)+'.dat')

        sentences  = read_data.get_sentences(data_file)
        K,encoding = read_data.sentences_to_word_encodings(sentences)
        print("K={}".format(K))

        task_func(NNClass, n_epochs, sentences, K, encoding)

if __name__ == '__main__':
    main()
