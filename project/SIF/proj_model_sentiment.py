import numpy as np
import theano
from theano import tensor as T
from theano import config
import lasagne

NUM_LABELS = 4

class proj_model_sentiment(object):

    def __init__(self, w2v, params):

        if params.npc > 0:
            pc = theano.shared(np.asarray(params.pc, dtype = config.floatX))

        vector_input = T.matrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, 300))
        l_out = lasagne.layers.DenseLayer(l_in, params.layersize, nonlinearity=params.nonlinearity)
        embg = lasagne.layers.get_output(l_out, {l_in:vector_input})
        if params.npc <= 0:
            print("#pc <=0, do not remove pc")
        elif params.npc == 1:
            print("#pc == 1")
            proj =  embg.dot(pc.transpose())
            embg = embg - theano.tensor.outer(proj, pc)
        else:
            print("#pc > 1")
            proj =  embg.dot(pc.transpose())
            embg = embg - theano.tensor.dot(proj, pc)

        #########

        l_in2 = lasagne.layers.InputLayer((None, params.layersize))
        l_sigmoid = lasagne.layers.DenseLayer(l_in2, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        l_softmax = lasagne.layers.DenseLayer(l_sigmoid, NUM_LABELS, nonlinearity=T.nnet.softmax)
        X = lasagne.layers.get_output(l_softmax, {l_in2:embg})
        cost = T.nnet.categorical_crossentropy(X,scores)
        prediction = (T.argmax(X, axis=1), T.max(X, axis=1))

        self.all_params = lasagne.layers.get_all_params(l_out, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)


        self.trainable = self.all_params
        cost = T.mean(cost)

        self.feedforward_function = theano.function([vector_input], embg)
        self.scoring_function = theano.function([vector_input],prediction)
        self.cost_function = theano.function([scores, vector_input], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, vector_input], cost, updates=updates)
