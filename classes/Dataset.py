import os
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from classes.Utils import *

class VCTKDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000):
        super(VCTKDataset, self).__init__()
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.audio_dir = os.path.join(root_dir, "wav48_silence_trimmed/")
        self.speaker_folders = sorted(os.listdir(self.audio_dir))
        self.num_speakers = len(self.speaker_folders)

        # mapping entre identifiant speaker vers un indice
        self.speaker_to_idx = {speaker: i for i, speaker in enumerate(self.speaker_folders)}

        # paires (audio_path, speaker_id)
        self.data = []
        for speaker in self.speaker_folders:
            speaker_path = os.path.join(self.audio_dir, speaker)
            if os.path.isdir(speaker_path):
                for file in sorted(os.listdir(speaker_path)): # sorted() pas obligé, ça me facilitait la tâche quand je débuggais d'avoir les fichiers dans l'ordre
                    if file.endswith(".flac"):
                        audio_path = os.path.join(speaker_path, file)
                        self.data.append((audio_path, speaker))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """
        Returns:
            - waveform (Tensor): The speech audio as a tensor.
            - speaker_id (Tensor): The numerical speaker ID.
        """
        audio_path, speaker = self.data[idx]
        speaker_id = self.speaker_to_idx[speaker]
        
        waveform, sample_r = torchaudio.load(audio_path)
        
        # resample si besoin (tout caler sur 16kHz)
        if sample_r != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_r, self.sample_rate)(waveform)
            
        speaker_conditioning = F.one_hot(torch.tensor(speaker_id), num_classes=self.num_speakers).float()

        return waveform, speaker_conditioning

def collate_fn(batch):
    """
    Custom collate function to pad variable-length audio waveforms.
    """
    waveforms, speaker_ids = zip(*batch)

    max_length = max(w.shape[1] for w in waveforms)

    # pad tous les samples jusque max_length
    padded_waveforms = torch.stack([F.pad(w, (0, max_length - w.shape[1])) for w in waveforms])
    # quantize (mu-law)
    quantized_waveforms = quantize(padded_waveforms, 8)
    
    speaker_ids = torch.stack(speaker_ids)
    
    return quantized_waveforms, speaker_ids