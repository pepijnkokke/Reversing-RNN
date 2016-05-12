import theano
import theano.tensor as T


def encoder_decoder(encoder_size,decoder_size,embed_size,lexicon_size):


    # Embedding matrix, shared between encoder and decoder.
    E = theano.shared(np.random.uniform(
        size=(embed_size, lexicon_size), low=-0.1, high=0.1))


    def encoder():
        """ Construct the encoder. """

        x  = T.vector()

        # State of the hidden layer.
        h  = theano.shared(np.zeros(encoder_size))

        # Weigts of arrows from the input x to the candidate
        # state h', the update gate z, and the reset gate r.
        W  = theano.shared(np.random.uniform(
            size=(encoder_size, embed_size), low=-0.1, high=0.1))
        Wz = theano.shared(np.random.uniform(
            size=(encoder_size, embed_size), low=-0.1, high=0.1))
        Wr = theano.shared(np.random.uniform(
            size=(encoder_size, embed_size), low=-0.1, high=0.1))

        # Weigts of arrows from the hidden state h to the candidate
        # state h', the update gate z, and the reset gate r.
        U  = theano.shared(np.random.uniform(
            size=(encoder_size, encoder_size), low=-0.1, high=0.1))
        Uz = theano.shared(np.random.uniform(
            size=(encoder_size, encoder_size), low=-0.1, high=0.1))
        Ur = theano.shared(np.random.uniform(
            size=(encoder_size, encoder_size), low=-0.1, high=0.1))

        # Weights of the arrows from the hidden state h to the output.
        V  = theano.shared(np.eye(encoder_size))


        def next_hidden_state(x, h_prev):
            """ Encode a word / Compute the next value of the hidden state. """

            z      = T.nnet.sigmoid(Wz.dot(E.dot(x)) + Uz.dot(h_prev))
            r      = T.nnet.sigmoid(Wr.dot(E.dot(x)) + Ur.dot(h_prev))
            h_cand = T.tanh(W.dot(E.dot(x)) + U.dot(r * h_prev))
            h_curr = (z * h_prev) + ((1 - z) * h_cand)

            return h_curr

        def encode_phrase():
            """ Encode a phrase / Repeatedly compute the next value of the hidden state. """

            words = T.matrix()

            results, _ = theano.scan(next_hidden_state, outputs_info=h, sequences=[words])

            updates = [(h,results[-1])]
            outputs = [T.tanh(V.dot(results[-1]))]

            return theano.function(inputs, outputs, updates=updates)


        encode_phrase = encode_phrase()
        return encode_phrase



    def decoder():
        """ Construct the decoder. """

        c  = T.vector()
        y  = T.vector()

        # State of the hidden layer.
        h  = theano.shared(np.zeros(decoder_size))

        # Weigts of arrows from the input x to the candidate
        # state h', the update gate z, and the reset gate r.
        W  = theano.shared(np.random.uniform(
            size=(decoder_size, embed_size), low=-0.1, high=0.1))
        Wz = theano.shared(np.random.uniform(
            size=(decoder_size, embed_size), low=-0.1, high=0.1))
        Wr = theano.shared(np.random.uniform(
            size=(decoder_size, embed_size), low=-0.1, high=0.1))

        # Weigts of arrows from the hidden state h to the candidate
        # state h', the update gate z, and the reset gate r.
        U  = theano.shared(np.random.uniform(
            size=(decoder_size, decoder_size), low=-0.1, high=0.1))
        Uz = theano.shared(np.random.uniform(
            size=(decoder_size, decoder_size), low=-0.1, high=0.1))
        Ur = theano.shared(np.random.uniform(
            size=(decoder_size, decoder_size), low=-0.1, high=0.1))

        C  = theano.shared(np.random.uniform(
            size=(decoder_size, encoder_size), low=-0.1, high=0.1))
        Cz = theano.shared(np.random.uniform(
            size=(decoder_size, encoder_size), low=-0.1, high=0.1))
        Cr = theano.shared(np.random.uniform(
            size=(decoder_size, encoder_size), low=-0.1, high=0.1))

        Oc = theano.shared(np.random.uniform(
            size=(decoder_size, encoder_size), low=-0.1, high=0.1))
        Oh = theano.shared(np.random.uniform(
            size=(decoder_size, decoder_size), low=-0.1, high=0.1))
        Oy = theano.shared(np.random.uniform(
            size=(decoder_size, lexicon_size), low=-0.1, high=0.1))

        Gl = theano.shared(np.random.uniform(
            size=(lexicon_size, embed_size), low=-0.1, high=0.1))
        Gr = theano.shared(np.random.uniform(
            size=(embed_size, decoder_size), low=-0.1, high=0.1))

        # Weights of the arrows from the hidden state h to the output.
        V  = theano.shared(np.eye(decoder_size))

        def next_hidden_state(y_prev, h_prev):
            """
            Compute the next value of the hidden state.
            """

            z      = T.nnet.sigmoid(Wz.dot(E.dot(y_prev)) + Uz.dot(h_prev) + Cz.dot(c))
            r      = T.nnet.sigmoid(Wr.dot(E.dot(y_prev)) + Ur.dot(h_prev) + Cr.dot(c))
            h_cand = T.tanh(W.dot(E.dot(y_prev)) + (r * (U.dot(h_prev) + C.dot(c))))
            h_curr = (z * h_prev) + ((1 - z) * h_cand)

            return h_curr

        def decode_word(y_prev,h_prev):
            """
            Compute the next value of the hidden state,
            and the output value for the words.
            """

            h_next = next_hidden_state(y_prev, h_prev)
            s      = T.nnet.softmax(
                Gl.dot(Gr.dot(Oh.dot(h_next) + Oy.dot(y_prev) + Oc.dot(c))))

            return (h_next,s)

        # Durnig training, we know the desired output phrase, and
        # therefore we can write a scan over the output phrase.
        # This scan will be feeding the _desired_ output word, at each
        # position, into the decode_word call at the next position.

        # During testing, however, it becomes a lot more involved to
        # write this as a scan. Therefore, we should perhaps simply
        # call the decode_word function, as a theano function, from
        # Python and stop whenever we generate the end_of_sentence
        # token.
