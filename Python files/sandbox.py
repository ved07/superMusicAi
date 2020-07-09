from midi2audio import
open('output.wav', 'a+')

FluidSynth().midi_to_audio('test_output.mid', 'output.wav')