import theano
import theano.tensor as T


def decoder(input_size, hidden_size, output_size, embed_size):
    """
    input_size:  Equal to the hidden_size of the encoder.
    hidden_size: The hidden_size of the decoder.
    output_size: The size of a vector over the lexicon.
    """

    y  = T.vector()

    # State of the hidden layer.
    h  = theano.shared(np.zeros(hidden_size))

    # Weigts of arrows from the input x to the candidate
    # state h', the update gate z, and the reset gate r.
    W  = theano.shared(np.random.uniform(
        size=(hidden_size, embed_size), low=-0.1, high=0.1))
    Wz = theano.shared(np.random.uniform(
        size=(hidden_size, embed_size), low=-0.1, high=0.1))
    Wr = theano.shared(np.random.uniform(
        size=(hidden_size, embed_size), low=-0.1, high=0.1))

    # Weigts of arrows from the hidden state h to the candidate
    # state h', the update gate z, and the reset gate r.
    U  = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))
    Uz = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))
    Ur = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))

    C  = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))
    Cz = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))
    Cr = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))

    Oc = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))
    Oh = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))
    Oy = theano.shared(np.random.uniform(
        size=(hidden_size, output_size), low=-0.1, high=0.1))

    E  = theano.shared(np.random.uniform(
        size=(embed_size, output_size), low=-0.1, high=0.1))
    Gl = theano.shared(np.random.uniform(
        size=(output_size, embed_size), low=-0.1, high=0.1))
    Gl = theano.shared(np.random.uniform(
        size=(embed_size, hidden_size), low=-0.1, high=0.1))


def encoder(input_size, hidden_size):
    """
    input_size:  The size of a vector over the lexicon.
    hidden_size: The hidden_size of the encoder.
    """

    x  = T.vector()

    # State of the hidden layer.
    h  = theano.shared(np.zeros(hidden_size))

    # Weigts of arrows from the input x to the candidate
    # state h', the update gate z, and the reset gate r.
    W  = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))
    Wz = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))
    Wr = theano.shared(np.random.uniform(
        size=(hidden_size, input_size), low=-0.1, high=0.1))

    # Weigts of arrows from the hidden state h to the candidate
    # state h', the update gate z, and the reset gate r.
    U  = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))
    Uz = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))
    Ur = theano.shared(np.random.uniform(
        size=(hidden_size, hidden_size), low=-0.1, high=0.1))

    # Weights of the arrows from the hidden state h to the output.
    V  = theano.shared(np.eye(hidden_size))

    def encode_word(x, h_prev):

        z      = T.nnet.sigmoid(Wz.dot(x) + Uz.dot(h_prev))
        r      = T.nnet.sigmoid(Wr.dot(x) + Ur.dot(h_prev))
        h_cand = T.tanh(W.dot(x) + U.dot(r * h_prev))
        h_curr = (z * h_prev) + ((1 - z) * h_cand)

        return h_curr

    def encode_phrase():

        words = T.matrix()

        results, _ = theano.scan(encode_word, outputs_info=h, sequences=[words])

        updates = [(h,results[-1])]
        outputs = [T.tanh(V.dot(results[-1]))]

        return theano.function(inputs, outputs, updates=updates)

    encode_phrase = encode_phrase()
