# laughdetector/load.py
import json
import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class LaughDataset(Dataset):
    def __init__(self, audio_dir, timestamps_file, window_s=2.0, hop_s=1.0,
                 n_mfcc=40, n_fft=2048, hop_length_feature=512):
        self.audio_dir = audio_dir
        self.window_s = window_s
        self.hop_s = hop_s
        self.n_mfcc = n_mfcc # Store these for dataset info

        # Pass feature parameters to _prepare_segments
        self.segments = self._prepare_segments(timestamps_file, n_mfcc, n_fft, hop_length_feature)

        # Store sample rate if it's consistent for all audio (used for feature transform setup)
        if self.segments:
            first_desc = json.load(open(timestamps_file))[0]
            first_audio_path = os.path.join(self.audio_dir, first_desc["filename"])
            try:
                metadata = torchaudio.info(first_audio_path)
                self.sr = metadata.sample_rate
            except Exception as e:
                print(f"Error getting info for first audio file {first_audio_path} to determine sample rate: {e}")
                self.sr = None
        else:
            self.sr = None


    def _prepare_segments(self, timestamps_file, n_mfcc, n_fft, hop_length_feature):
        items = json.load(open(timestamps_file))
        all_segments = []

        if not items:
            print("No items found in timestamps file. Dataset will be empty.")
            return []

        first_desc = items[0]
        first_audio_path = os.path.join(self.audio_dir, first_desc["filename"])
        try:
            metadata = torchaudio.info(first_audio_path)
            sample_rate = metadata.sample_rate
        except Exception as e:
            print(f"Error getting info for first audio file {first_audio_path} to determine sample rate: {e}")
            return []

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length_feature, "n_mels": n_mfcc * 2}
        )
        self.amplitude_to_db = T.AmplitudeToDB()


        W_audio = int(self.window_s * sample_rate)
        H_audio = int(self.hop_s * sample_rate)

        if W_audio <= 0:
            raise ValueError(f"Window size (W_audio) is {W_audio}. It must be greater than 0. Check window_s and sample rate.")

        for desc in items:
            path = os.path.join(self.audio_dir, desc["filename"])
            try:
                audio, sr_actual = torchaudio.load(path)
            except Exception as e:
                print(f"Error loading audio file {path}: {e}. Skipping.")
                continue

            if sr_actual != sample_rate:
                # Resample if sample rates don't match
                resampler = T.Resample(orig_freq=sr_actual, new_freq=sample_rate)
                audio = resampler(audio)


            T_audio_samples = audio.size(1)

            if audio.size(0) > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            targets_audio_samples = torch.ones(T_audio_samples, dtype=torch.long)
            for s_sec, e_sec in desc["timestamps"]:
                s_audio, e_audio = int(s_sec * sample_rate), int(e_sec * sample_rate)
                s_audio = max(0, s_audio)
                e_audio = min(T_audio_samples, e_audio)
                if s_audio < e_audio:
                    targets_audio_samples[s_audio:e_audio] = 0

            for start_audio in range(0, T_audio_samples - W_audio + 1, H_audio):
                end_audio = start_audio + W_audio
                audio_segment_raw = audio[:, start_audio:end_audio]

                if audio_segment_raw.size(1) == W_audio:
                    # Apply MFCC transform
                    features = self.mfcc_transform(audio_segment_raw)
                    features = self.amplitude_to_db(features)

                    num_feature_frames = features.size(2)
                    seg_targets_features = torch.zeros(num_feature_frames, dtype=torch.long)
                    for i in range(num_feature_frames):
                        frame_start_sample = start_audio + i * hop_length_feature
                        frame_end_sample = min(start_audio + (i + 1) * hop_length_feature, end_audio)
                        if torch.any(targets_audio_samples[frame_start_sample:frame_end_sample] == 0):
                            seg_targets_features[i] = 0 # Mark as laugh
                        else:
                            seg_targets_features[i] = 1 # Mark as non-laugh

                    all_segments.append((features.squeeze(0), seg_targets_features))
                else:
                    pass
        return all_segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment, targets = self.segments[idx]
        return segment, targets


def load(audio_dir, timestamps_file, window_s=2.0, hop_s=1.0,
         n_mfcc=40, n_fft=2048, hop_length_feature=512):
    return LaughDataset(audio_dir, timestamps_file, window_s, hop_s,
                        n_mfcc=n_mfcc, n_fft=n_fft, hop_length_feature=hop_length_feature)