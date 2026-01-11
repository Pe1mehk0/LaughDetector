import os
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio import save
import tempfile  # More reliable than hardcoded /tmp/

try:
    from playsound import playsound
except ImportError:
    playsound = None


def play_tensor(tensor, freq=44_100):
    if playsound is None:
        print("playsound not installed. Skipping playback.")
        return

    tmp_dir = tempfile.gettempdir()
    _tmp_path = os.path.join(tmp_dir, 'tmp-tensor.wav')

    if len(tensor.size()) == 1:
        tensor = tensor.unsqueeze(0)

    save(_tmp_path, tensor.cpu(), freq)
    playsound(_tmp_path)
    os.remove(_tmp_path)


def prepare_sequence(audio):
    return audio.reshape(1, *audio.size())


def postprocess(model_out: torch.Tensor, mfcc_frame_rate_hz: float, thr: float = 0.5,
                min_len: float = 0.6, max_gap: float = 0.4):

    probs = torch.softmax(model_out, dim=1)[:, 0]
    mask = probs > thr

    min_frames = int(min_len * mfcc_frame_rate_hz)
    max_gap_frames = int(max_gap * mfcc_frame_rate_hz)

    # Step 1: Segmentation
    initial_segments = []
    start_f = None
    for i, m in enumerate(mask):
        if m and start_f is None:
            start_f = i
        elif not m and start_f is not None:
            initial_segments.append((start_f, i - 1))
            start_f = None
    if start_f is not None:
        initial_segments.append((start_f, len(mask) - 1))

    # Step 2: Length Filter
    filtered = [s for s in initial_segments if (s[1] - s[0] + 1) >= min_frames]

    # Step 3: Merge Gaps
    if not filtered: return []
    merged = []
    curr_s, curr_e = filtered[0]
    for i in range(1, len(filtered)):
        next_s, next_e = filtered[i]
        if (next_s - (curr_e + 1)) <= max_gap_frames:
            curr_e = next_e
        else:
            merged.append((curr_s, curr_e))
            curr_s, curr_e = next_s, next_e
    merged.append((curr_s, curr_e))

    return [(s / mfcc_frame_rate_hz, e / mfcc_frame_rate_hz) for s, e in merged]


def split_one(audio, time_mark):
    return audio[:, time_mark[0]:time_mark[1] + 1]


def split_audio(audio, time_marks):
    return [split_one(audio, time_mark) for time_mark in time_marks]