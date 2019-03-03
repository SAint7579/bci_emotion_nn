import os
#Using tensorflow on cpu
os.environ['CUDA_VISIBLE_DEVICES'] ='-1'
import tensorflow as tf

def create_LSTM_cell(hidden_states,drop_out = False, keep_prob = None):
    '''
        creates a basic LSTM cell
        Args:
        hidden states (int) : Number of hidden units in the cell
        drop_out (bool) : If dropout layer is required (Default: False)
        keep_prob (tf.placeholder) : Holding probability for the dropout layer (Default: None)
        Return:
        Basic LSTM cell
    '''
    #Creating the LSTM cell
    cell = tf.contrib.rnn.BasicLSTMcell(num_units = hidden_states, activation = tf.nn.elu)
    #Adding dropout layer
    if drop_out == True:
        cell = tf.contrib.rnn.DropoutWrapper(cell,keep_prob = keep_prob)
    return cell

def create_Stacked_cell(cells):
    '''
        creates a basic Multi/Stacked cell structure
        Args:
        cells(list) : List for cells that are to be stacked
        Return:
        MultiRNNcell
    '''
    #Creating the MultiRNN cell
    cell = tf.contrib.rnn.MultiRNNcell(cells = cells)
    #Wrapping the cell    
    return cell

def output_wrapper(n_outputs, cell):
    '''
        Wraps the rnn cell's output to the given value
        Args:
        n_outputs (int) : Number of output nodes
        cells(list) : List for cells that are to be stacked
        Return:
        Wrapped RNNcell
    '''
    wrapped = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = n_outputs)
    return wrapped

def dropout_wrapper(keep_prob, cell):
    '''
        Wraps the rnn cell's neurons in a dropout layer
        Args:
        keep_proba(tf.placeholder) : Probability of keeping a node's output
        cells(list) : List for cells that are to be stacked
        Return:
        Wrapped RNNcell
    '''
    wrapped = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return wrapped


def create_RNN(cell,inp):
    '''
    Creates a neural network using the provided cell
    Args:
    cell (tf.contrib.rnn) : The cell with which the rnn is to be created
    inp (tf.placeholder) : The input set given to the RNN
    Returns:
    output: Output generator of the neural network
    state: Final state of the neural network
    '''
    return tf.nn.dynamic_rnn(cell,inputs = inp, dtype=tf.float32)
