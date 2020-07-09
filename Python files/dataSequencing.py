import numpy as np
import musicParsing
from keras.utils import np_utils


class DataStructuring:
    def __init__(self, sequenceLength):
        parsedNotes = musicParsing.Parse()
        self.notes = parsedNotes.returnNotes('dataset')
        print(self.notes)
        self.networkInput = []
        self.networkOutput = []
        self.sequenceLength = sequenceLength

    def LengthOfNotes(self):
        n_vocab = len(set(self.notes))
        return n_vocab

    def sequenceData(self):
        notes = self.notes
        n_vocab = len(set(self.notes))
        """ Prepare the sequences used by the Neural Network """
        sequence_length = 100
        print(notes)
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        print(pitchnames)
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        print(network_input.shape)
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = np_utils.to_categorical(network_output)

        return (network_input, network_output)

    def prepareSequencesTrain(self):
        """ Prepare the sequences used by the Neural Network """
        # map between notes and integers and back
        self.pitchnames = sorted(set(item for item in self.notes))
        note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))

        sequence_length = 100
        network_input = []
        output = []
        for i in range(0, len(self.notes) - sequence_length, 1):
            sequence_in = self.notes[i:i + sequence_length]
            sequence_out = self.notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        normalized_input = normalized_input / float(len(set(self.notes)))

        return (network_input, normalized_input)





