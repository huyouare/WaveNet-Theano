import os, path
import glob
import numpy as np
import audioop
import wave

# Default values
input_num_channels = 1
input_sample_rate = 48000
input_sample_width = 2

SAMPLE_RATE = 24000     # Output/test data sample rate
Q_FACTOR = 1            # Additional linear quantization (for testing only)
LIMIT = 20              # Number of files
MIN_DURATION = 4.0      # Minimum duration in seconds

def generate_input_data():
    """
    Outputs:
    raw_data
    train_data
    """

    print "=======================================\n" + \
          "=         GENERATING INPUT DATA       =\n" + \
          "======================================="

    input_data = np.zeros([LIMIT, int(MIN_DURATION * SAMPLE_RATE)])
    count = 0
    for filename in glob.glob("VCTK-Corpus/wav48/p225/*.wav"):
        data = open_input(filename)
        if len(data) / float(SAMPLE_RATE) > MIN_DURATION:
            input_data[count, :] = data[:int(MIN_DURATION * SAMPLE_RATE)]
            count += 1
        if count >= LIMIT: break

    train_data = input_data[:count, :]
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    raw_data = train_data.astype('int64')

    train_data = (train_data - 128) / 128.
    train_data = train_data.astype('float32')

    print "=======================================\n" + \
          "=       DATA GENERATION FINISHED      =\n" + \
          "======================================="

    return raw_data, train_data

def open_input(filename):
    stream = wave.open(filename,"rb")

    input_num_channels = stream.getnchannels()
    input_sample_rate = stream.getframerate()
    input_sample_width = stream.getsampwidth()
    input_num_frames = stream.getnframes()

    raw_data = stream.readframes(input_num_frames) # Returns byte data
    stream.close()

    total_samples = input_num_frames * input_num_channels

    print "Sample Width: {} ({}-bit)".format(input_sample_width, 8 * input_sample_width)
    print "Number of Channels: " + str(input_num_channels)
    print "Sample Rate " + str(input_sample_rate)

    print "Number of Samples: " + str(total_samples)
    print "Duration: {0:.2f}s".format(total_samples / float(input_sample_rate))
    print "Raw Data Size: " + str(len(raw_data))

    if input_sample_rate != SAMPLE_RATE:
        u_law = audioop.ratecv(raw_data, input_sample_width, input_num_channels, input_sample_rate, SAMPLE_RATE, None)
        u_law = audioop.lin2ulaw(u_law[0], input_sample_width)
    else:
        u_law = audioop.lin2ulaw(raw_data, input_sample_width)

    u_law = list(u_law)
    u_law = [ord(x)//Q_FACTOR for x in u_law]

    return np.asarray(u_law)

def save_output(data, filename):
    # data is the u-law quantized sample
    u_law = data
    u_law = [chr(x) for x in u_law]
    u_law = ''.join(u_law)

    original = audioop.ulaw2lin(u_law, input_sample_width)
    print "output data size: " + str(len(original))

    output = wave.open(filename,'w')
    output.setparams((input_num_channels, input_sample_width, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
    output.writeframes(original)
    output.close()
