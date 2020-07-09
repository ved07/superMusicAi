from music21 import converter, instrument, note, chord
import glob
import numpy as np
import pickle

class Parse():
    def __init__(self):
        self.notes = []

    def returnNotes(self,filename):
        notes = []

        for file in glob.glob("dataset/*.mid"):
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try:  # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                # print(element)
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        with open('data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

        return notes
