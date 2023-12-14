import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import librosa
from BeatNet.model import BDA
from madmom.features import DBNDownBeatTrackingProcessor
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from BeatNet.log_spect import LOG_SPECT

import soundfile as sf
import argparse

BEATNET_PATH = "/home/greg/Workshop-Generative-AI/genmus_bcn/nmf_morph_dafx2020_python/BeatNet"

class Beat_detector:
    
    def __init__(self, model, trained_model_dir, mode='offline', inference_model='DBN', sample_rate=22050, device='cpu'):
        self.model = model
        self.mode = mode
        self.inference_model = inference_model
        self.device = device
        if sample_rate != 22050:
            raise RuntimeError('Sample rate should be 22050')
        self.sample_rate = sample_rate
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)
        if self.inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=self.plot, mode=self.mode)
        elif self.inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')
        
        # Create and load model
        self.model = BDA(272, 150, 2, self.device)   #Beat Downbeat Activation detector
        #loading the pre-trained BeatNet CRNN weigths
        if model == 1:  # GTZAN out trained model
            self.model.load_state_dict(torch.load(os.path.join(trained_model_dir, 'models/model_1_weights.pt')), strict=False)
        elif model == 2:  # Ballroom out trained model
            self.model.load_state_dict(torch.load(os.path.join(trained_model_dir, 'models/model_2_weights.pt')), strict=False)
        elif model == 3:  # Rock_corpus out trained model
            self.model.load_state_dict(torch.load(os.path.join(trained_model_dir, 'models/model_3_weights.pt')), strict=False)
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.eval()
        
    def process(self, audio):
        if self.inference_model != "DBN":
            raise RuntimeError('The infernece model should be set to "DBN" for the offline mode!')
        if audio.any():
            preds = self.activation_extractor(audio)    # Using BeatNet causal Neural network to extract activations
            output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
            return output
        
        else:
            raise RuntimeError('No audio!')
    
    def activation_extractor(self, audio):
        with torch.no_grad():
            feats = self.proc.process_audio(audio).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0).to(self.device)
            preds = self.model(feats)[0]  # extracting the activations by passing the feature through the NN
            preds = self.model.final_pred(preds)
            preds = preds.cpu().detach().numpy()
            preds = np.transpose(preds[:2, :])
        return preds

class Song_concatenator:
    
    def __init__(self, beat_detector, seconds=3):
        self.beat_detector = beat_detector
        self.seconds = seconds
        
    def concatenate(self, song1, song2, sr):
        beats_1 = self.beat_detector.process(song1)
        beats_2 = self.beat_detector.process(song2)
        _, last_beat = self._find_last_downbeat(beats_1, len(song1)/sr)
        _, first_beat = self._find_first_downbeat(beats_2)
        song1_cut = song1[:int(last_beat[0]*sr)]
        song2_cut = song2[int(first_beat[0]*sr):]
        concatenated_song = np.concatenate((song1_cut, song2_cut))
        return concatenated_song, last_beat, first_beat
    
    def _find_last_downbeat(self, beats, audio_duration):
        for i in range(len(beats)-1,0,-1):
            if beats[i][1] == 1. and beats[i][0] < audio_duration-self.seconds:
                return i, beats[i]
        raise RuntimeError(f'First song: downbeat not found!')
      
    def _find_first_downbeat(self, beats):
        for i in range(len(beats)):
            if beats[i][1] == 1. and beats[i][0] > self.seconds:
                return i, beats[i]
        raise RuntimeError(f'Second song: downbeat not found!')

def song_concatenation(file1, file2, save_path=None, verbose=False):
    #file="house120.mp3"
    #file2="2house120.mp3"
    
    audio1, sr = librosa.load(file1)
    audio2, _ = librosa.load(file2)
          
    beatnet = Beat_detector(1, BEATNET_PATH, "offline", 'DBN', sample_rate=sr)
    song_concatenator = Song_concatenator(beatnet, 3)
    
    concatenated_song, last_beat, first_beat = song_concatenator.concatenate(audio1, audio2, sr)
        
    if save_path is not None:
        sf.write(save_path, concatenated_song, sr,'PCM_24')
        
    if verbose:
        beats_1 = song_concatenator.beat_detector.process(audio1)
        beats_2 = song_concatenator.beat_detector.process(audio2)

        print("First Song: Beats/Downbeats (offline): ", beats_1)
        print("Second Song: Beats/Downbeats (offline): ", beats_2)
    
        print(f"Last downbeat of the first song: {last_beat}")
        print(f"First downbeat of the second song: {first_beat}")
        
        plt.plot(audio1)
        for b_off in beats_1: 
            if b_off[1] == 1.:
                color = 'red'
            else: 
                color = 'green'
        plt.axvline(x=b_off[0]*sr, color=color)
        plt.xlabel('Time (samples)')
        plt.title("First Song: audio waveform and the estimated beat positions")
        plt.show()
    
        plt.plot(audio2)
        for b_off in beats_2: 
            if b_off[1] == 1.:
                color = 'red'
            else: 
                color = 'green'
        plt.axvline(x=b_off[0]*sr, color=color)
        plt.xlabel('Time (samples)')
        plt.title("Second song: audio waveform and the estimated beat positions")
        plt.show()
        
        plt.plot(concatenated_song)
        plt.xlabel('Time (samples)')
        plt.title("Concatenated song")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPM calculator script")
    parser.add_argument('--first_song', type=str, help='Path to the first song', default="test/house120.mp3")
    parser.add_argument('--second_song', type=str, help='Path to the second song', default="test/2house120.mp3")
    parser.add_argument('--save_path', type=str, help='Path where to save concatenated song', default="test/concatenated_song.wav")
    parser.add_argument('--verbose', type=bool, help='Whether to print the output', default=False)
    args = parser.parse_args()
    print("Concatenating...")
    song_concatenation(args.first_song, args.second_song, args.save_path, args.verbose)
    print("Song concatenated successfully!")