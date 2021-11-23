import argparse
import parse
import numpy as np
import crepe
import librosa
import soundfile
import pytsmod as tsm
import pydub
import os
import sys

def linear(t, initial: float, target: float):
    if(t.shape[0] == 0):
        return t
    x = np.linspace(0, 1, num=t.shape[0])
    return (x * (target - initial)) + initial

def ADSR(t, attack, decay, sustain, release):
    if t.shape[0] == 0:
        return t, np.array([])

    adsr = np.zeros_like(t)

    t_base = t - np.min(t)
    attack_idx = np.searchsorted(t_base, attack)
    decay_idx = np.searchsorted(t_base, attack+decay)

    adsr[:attack_idx] = linear(t_base[:attack_idx], 0, 1)
    adsr[attack_idx:decay_idx] = linear(
            t_base[attack_idx:decay_idx], 1, sustain)
    adsr[decay_idx:] = sustain

    release = None

    return adsr, release

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", dest="input", type=str, default=None, required=True, help="Path to 'voice' file")
    parser.add_argument("-m", dest="MIDI", type=str, required=True, help="Path to MIDI file")

    args = parser.parse_args()

    input_file = args.input
    midi_file = args.MIDI

    midi_file = parse.Parser(midi_file)

    y, sr = librosa.core.load(input_file)

    x, trim_idx = librosa.effects.trim(y, top_db=10)
    
    soundfile.write("test.wav", x, sr)
    _, f0_crepe, _, _ = crepe.predict(x, sr, viterbi=True, step_size=10)

    def to_seconds(tick, ticks_per_quarter, bpm):
        return ((tick * 60) / (bpm * ticks_per_quarter))


    def play(tracks, ticks_per_quarter, bpm):
        if os.path.isfile("out.wav"):
            os.remove("out.wav")


        print(len(tracks))

        duration = librosa.get_duration(x, sr)
        
        for track in range(0, len(tracks)):

            if(len(tracks[track]) == 0): continue

            song_time = to_seconds(tracks[track][-1][3], ticks_per_quarter, bpm)

            t = np.linspace(0, song_time, (int)(song_time * 44100), False)

            song = np.zeros_like(t)
            num_of_notes = 0
            for note in tracks[track]:
                num_of_notes+=1
                sys.stdout.write("\033[2K\033[1G"), print(f"{num_of_notes} out of {len(tracks[track])}", end="")
                start_pos = np.searchsorted(t, to_seconds(note[2], ticks_per_quarter, bpm))
                end_pos = np.searchsorted(t, to_seconds(note[3], ticks_per_quarter, bpm))
                
                note_t = t[start_pos:end_pos]
    
                envelope, release = ADSR(note_t, 0.5, 0.2, 1, 0.2)

                target = to_seconds(note[3], ticks_per_quarter, bpm) - to_seconds(note[2], ticks_per_quarter, bpm)
                #print(note[0][0])
                if(note[0][0]-60 < 0):
                    temp = (tsm.tdpsola(x, sr, f0_crepe, alpha=(target/duration), beta=pow(2, (note[0][0]-60)/12), p_hop_size=441, p_win_size=1470) * 0.25)
                else:
                    temp = (tsm.tdpsola(x, sr, f0_crepe, alpha=(target/duration), beta=pow(2, (note[0][0]-60)/12), p_hop_size=441, p_win_size=1470) * 1.25)
                #print(temp.shape[0])
                envelope, release = ADSR(note_t, 0.01, 0.2, 1.0, 0.2)
                #print(song[start_pos:end_pos].shape[0])
                temp = np.concatenate((temp, np.zeros(song[start_pos:end_pos].shape[0]-temp.shape[0])), axis=None)
                temp *= envelope
                #print(temp.shape)
                #print(song[start_pos:end_pos].shape)
                song[start_pos:end_pos] += temp

            soundfile.write(f"{track}.wav", song, 44100) 
        
        if os.path.isfile("0.wav"):
            final = pydub.AudioSegment.from_wav("0.wav")
        else:
            final = pydub.AudioSegment.from_wav("1.wav")
        for track in range(1, len(tracks)):
            if os.path.isfile(f"{track}.wav"):
                temp = pydub.AudioSegment.from_wav(f"{track}.wav")
                temp2 = final.overlay(temp, position=0)
                final = temp2

        final.export("out.wav", format="wav")


    play(*midi_file.parse())



