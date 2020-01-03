import torch
import torch.utils.data
import random
import numpy as np
from .symbols import ph2id, sp2id
from torch.utils.data import DataLoader
import os

class TextMelIDLoader(torch.utils.data.Dataset):
    
    def __init__(self, list_file, mean_std_file, shuffle=True):
        '''
        list_file: 1-column: /path/speaker_id/speaker_id_utt_id
        where the following exist:
        
        /path/speaker_id_utt_id.phones (from extract_features.py or phonemizer / festival)
        /path/speaker_id_utt_id.mel (from extract_features.py or deepvoice3 / festival)
        '''
        file_path_list = []
        with open(list_file) as f:
            lines = f.readlines()
            for path in lines:
                file_path_list.append(path)

        if shuffle:
            random.seed(1234)
            random.shuffle(file_path_list)
        
        self.file_path_list = file_path_list

    def get_text_mel_id_pair(self, path):
        '''
        You should Modify this function to read your own data.

        Returns:

        object: dimensionality
        -----------------------
        text_input: [len_text]
        mel: [mel_bin, len_mel]
        speaker_id: [1]
        ''' 
        # Deduce filenames
        path=path.rstrip()
        phones_path = path+".phones"
        mel_path = path+".mel.npy"
        speaker_id = path.split('/')[-2] # speaker id = dir in which files live

        # Load data from disk
        with open(phones_path, "r") as f:
            # the phoneme transcript should be one line, and space-delimited
            phones = [ ph2id[phone] for phone in f.read().split() ]
        mel = np.load(mel_path)
        
        # Format for pytorch
        phones = torch.LongTensor(phones)
        mel = torch.from_numpy(mel)
        print(mel.shape)
        speaker_id = torch.LongTensor([sp2id[speaker_id]])
        return phones, mel, speaker_id
            
    def __getitem__(self, index):
        return self.get_text_mel_id_pair(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


class TextMelIDCollate():

    def __init__(self, n_frames_per_step=2):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        '''
        batch is list of (text_input, mel, speaker_id)
        '''
        print(len(batch[0]))
            
        text_lengths = torch.IntTensor([len(x[0]) for x in batch])
        mel_lengths = torch.IntTensor([x[1].size(1) for x in batch])
        mel_bin = batch[0][1].size(0)

        max_text_len = torch.max(text_lengths).item()
        max_mel_len = torch.max(mel_lengths).item()
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        text_input_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)

        speaker_id = torch.LongTensor(len(batch))
        gate_padded = torch.FloatTensor(len(batch), max_mel_len)

        text_input_padded.zero_()
        mel_padded.zero_()
        speaker_id.zero_()
        gate_padded.zero_()

        for i in range(len(batch)):
            text =  batch[i][0]
            mel = batch[i][1]
            speaker_id[i] = batch[i][2][0]
            print("text=",text.shape,"mel=",mel.shape,"ID=",speaker_id.shape)

            text_input_padded[i,:text.size(0)] = text 
            mel_padded[i,  :, :mel.size(1)] = mel
            # make sure the downsampled gate_padded have the last eng flag 1. 
            gate_padded[i, mel.size(1)-self.n_frames_per_step:] = 1

        return text_input_padded, mel_padded, None, speaker_id, text_lengths, mel_lengths, gate_padded
    
