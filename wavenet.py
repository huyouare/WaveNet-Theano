"""

"""

import os, sys
sys.path.append(os.getcwd())

import numpy
numpy.random.seed(123)
import random
random.seed(123)

import theano
import theano.tensor as T
import lib
import lasagne
import scipy.misc

import time
import functools
import itertools

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams(seed=4884)

# Hyperparams
BATCH_SIZE = 10
DIM = 64        # Model dimensionality.
GRAD_CLIP = 1   # Elementwise grad clip threshold

# Dataset
N_CHANNELS = 1

# Other constants
TEST_BATCH_SIZE = 100   # batch size to use when evaluating on dev/test sets. This should be the max that can fit into GPU memory.
EVAL_DEV_COST = False   # whether to evaluate dev cost during training
GEN_SAMPLES = True      # whether to generate samples during training (generating samples takes WIDTH*HEIGHT*N_CHANNELS full passes through the net)
TRAIN_MODE = 'iters'    # 'iters' to use PRINT_ITERS and STOP_ITERS, 'time' to use PRINT_TIME and STOP_TIME
PRINT_ITERS = 10         # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 10000      # Stop after this many iterations
PRINT_TIME = 60*60      # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*2     # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)

DEPTH = 1               # number of dilated conv blocks
DILATION_LEVEL = 4      # maximum dilation will be 2^DILATION_LEVEL
LEARNING_RATE = 1e-2

lib.utils.print_model_settings(locals().copy())

def relu(x):
    # Using T.nnet.relu gives me NaNs. No idea why.
    return T.switch(x > lib.floatX(0), x, lib.floatX(0))

def DilatedConv1D(name, input_dim, output_dim, filter_size, inputs, dilation, mask_type=None, apply_biases=True):
    """
    inputs.shape: (batch size, length, input_dim)
    mask_type: None, 'a', 'b'
    output.shape: (batch size, length, output_dim)
    """
    def uniform(stdev, size):
        """uniform distribution with the given stdev and size"""
        return numpy.random.uniform(
            low=-stdev * numpy.sqrt(3),
            high=stdev * numpy.sqrt(3),
            size=size
        ).astype(theano.config.floatX)

    filters_init = uniform(
        1./numpy.sqrt(input_dim * filter_size),
        # output dim, input dim, height, width
        (output_dim, input_dim, filter_size, 1)
    )

    if mask_type is not None:
        filters_init *= lib.floatX(numpy.sqrt(2.))

    filters = lib.param(
        name+'.Filters',
        filters_init
    )

    if mask_type is not None:
        mask = numpy.ones(
            (output_dim, input_dim, filter_size, 1),
            dtype=theano.config.floatX
        )

        center = filter_size//2
        for i in xrange(filter_size):
            if (i > center):
                mask[:, :, i, :] = 0.
            # if (mask_type=='a' and i == center):
            #     mask[:, :, center] = 0.
        filters = filters * mask

    inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1, inputs.shape[2]))
    # conv2d takes inputs as (batch size, input channels, height[?], width[?])
    inputs = inputs.dimshuffle(0, 3, 1, 2)
    result = T.nnet.conv2d(inputs, filters, border_mode='half', filter_flip=False, filter_dilation=(dilation, 1))

    if apply_biases:
        biases = lib.param(
            name+'.Biases',
            numpy.zeros(output_dim, dtype=theano.config.floatX)
        )
        result = result + biases[None, :, None, None]

    result = result.dimshuffle(0, 2, 3, 1)
    return result.reshape((result.shape[0], result.shape[1], result.shape[3]))

def DilatedConvBlock(name, input_dim, output_dim, filter_size, inputs):
    result = inputs
    for i in xrange(DILATION_LEVEL):
        d = numpy.left_shift(2, i)
        result = DilatedConv1D(name+'Dilation'+str(d), DIM, DIM, 5, result, d, mask_type='b')
        result = relu(result)
    return result

def Softmax(output):
    # output.shape: (batch size, length, output_dim, 256)
    softmax_output = T.nnet.softmax(output.reshape((-1,output.shape[output.ndim-1])))
    return softmax_output.reshape(output.shape)

def sample_from_softmax(softmax_var):
    #softmax_var assumed to be of shape (batch_size, num_classes)
    old_shape = softmax_var.shape

    softmax_var_reshaped = softmax_var.reshape((-1,softmax_var.shape[softmax_var.ndim-1]))

    return T.argmax(
        T.cast(
            srng.multinomial(pvals=softmax_var_reshaped),
            theano.config.floatX
            ).reshape(old_shape),
        axis = softmax_var.ndim-1
        )

# inputs.shape: (batch size, length, input_dim)
inputs = T.tensor3('inputs') # normalized
raw_inputs = T.tensor3('raw_inputs').astype('int64')

output = DilatedConv1D('InputConv', N_CHANNELS, DIM, 5, inputs, 1, mask_type='a')

for i in xrange(DEPTH):
    output = DilatedConvBlock('DilatedConvBlock'+str(i), DIM, DIM, 5, output)
    output = relu(output)

output = DilatedConv1D('OutputConv1', DIM, DIM, 1, output, 1, mask_type='b')
output = relu(output)

output = DilatedConv1D('OutputConv2', DIM, DIM, 1, output, 1, mask_type='b')
output = relu(output)

output = DilatedConv1D('OutputConv3', DIM, 256, 1, output, 1, mask_type='b')

# output.shape: (batch size, length, output_dim)
output = Softmax(output.reshape((output.shape[0], output.shape[1], 1, output.shape[2])))

# categorical crossentropy
# coding dist shape: (batch_size * length * channels, 256)
# true dist shape: (batch_size * length * channels)
#                  symbolic vector of ints, each element represents
#                  position of '1' in one-hot encoding

cost = T.nnet.categorical_crossentropy(
    output.reshape((-1,output.shape[output.ndim - 1])),
    raw_inputs.reshape((raw_inputs.shape[0]*raw_inputs.shape[1],))
    ).mean()

debug_fn1 = theano.function(
        inputs=[inputs, raw_inputs], 
        outputs=output.reshape((-1,output.shape[output.ndim - 1])),
        on_unused_input='ignore' # to suppress unused input exeptions
)

debug_fn2 = theano.function(
        inputs=[inputs, raw_inputs], 
        outputs=raw_inputs.reshape((raw_inputs.shape[1],)),
        on_unused_input='ignore' # to suppress unused input exeptions
)

output = sample_from_softmax(output)

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.utils.print_params_info(params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
# grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=LEARNING_RATE)

train_fn = theano.function(
    [inputs, raw_inputs],
    cost,
    updates=updates,
    on_unused_input='warn'
)

eval_fn = theano.function(
    [inputs, raw_inputs],
    cost,
    on_unused_input='warn'
)

sample_fn = theano.function(
    [inputs],
    output,
    on_unused_input='warn'
)

raw_data, train_data = lib.wav.generate_input_data()
print raw_data.shape
print train_data.shape

print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0

numpy.set_printoptions(threshold=numpy.inf)

# sample_input = sample_input
# train_data = [(float(x) - 128.) / 128. for x in sample_input]
# train_data = numpy.asarray(train_data).astype('float32')
# # print(train_data)
# train_data = train_data.reshape(1, len(train_data), 1)
# sample_input = numpy.asarray(sample_input).astype('int64')
# # print(sample_input)
# sample_input = sample_input.reshape(1, len(sample_input), 1)
# print("raw" + str(sample_input.shape))

# output_test = debug_fn1(train_data, sample_input)
# print("output size " + str(output_test.shape))
# # print output_test
# output_test = debug_fn2(train_data, sample_input)
# print("output size 2 " + str(output_test.shape))
# print output_test[0:100]

for epoch in itertools.count():
    costs = []
    for x in xrange(10): # TODO: Change to batch size
        start_time = time.time()
        cost = train_fn(train_data, raw_data)
        total_time += time.time() - start_time
        total_iters += 1

        costs.append(cost)

        dev_costs = []
        print "epoch:{}\ttotal iters:{}\ttrain cost:{}\tdev cost:{}\ttotal time:{}\ttime per iter:{}".format(
                epoch,
                total_iters,
                numpy.mean(costs),
                numpy.mean(dev_costs),
                total_time,
                total_time / total_iters
            )

        if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
            (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME):

            dev_costs = []
            # if EVAL_DEV_COST:
            #     for images, targets in dev_data():
            #         images = images.reshape((-1, HEIGHT, WIDTH, 1))
            #         binarized = binarize(images)
            #         dev_cost = eval_fn(binarized)
            #         dev_costs.append(dev_cost)
            # else:
            #     dev_costs.append(0.)

            # print "epoch:{}\ttotal iters:{}\ttrain cost:{}\tdev cost:{}\ttotal time:{}\ttime per iter:{}".format(
            #     epoch,
            #     total_iters,
            #     numpy.mean(costs),
            #     numpy.mean(dev_costs),
            #     total_time,
            #     total_time / total_iters
            # )

            tag = "iters{}_time{}".format(total_iters, total_time)
            if GEN_SAMPLES:
                gen = sample_fn(train_data)
                print "Generated shape: " + str(gen.shape)
                gen = gen[0,:,:].reshape(gen.shape[1])
                print gen[0:100]
                lib.wav.save_output(gen, 'output/epoch{}_{}.wav'.format(epoch, tag))
            lib.save_params('params/params_{}.pkl'.format(tag))

            costs = []
            last_print_time += PRINT_TIME
            last_print_iters += PRINT_ITERS

        if (TRAIN_MODE=='iters' and total_iters == STOP_ITERS) or \
            (TRAIN_MODE=='time' and total_time >= STOP_TIME):

            print "Done!"

            sys.exit()
