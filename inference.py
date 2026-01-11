# inference.py
import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import click
from tqdm import tqdm

from laughdetector.nn.tagger import LSTMTagger
from laughdetector.utils import postprocess


MODEL_PATH = "./models/best_model_epoch_030_val_f1_0.9252.bin"

EMBEDDING_DIM = 40
HIDDEN_DIM = 64
TAGSET_SIZE = 2

N_FFT = 2048
HOP_LENGTH_FEATURE = 512
N_MELS = EMBEDDING_DIM * 2

WINDOW_S = 30.0
HOP_S = 1.0

THRESHOLD = 0.5

@click.command()
@click.option("--audio-dir", required=True, help="Directory containing audio files to process.")
@click.option("--output-dir", default="./inference_results", help="Directory to save laughter timestamps.")
@click.option("--model-path", default=MODEL_PATH, help="Path to the trained model.")
@click.option("--force-sr", type=int, default=48000, help="Force resampling of audio to this sample rate.")

def main(audio_dir, output_dir, model_path, force_sr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, TAGSET_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {model_path}")

    mfcc_transform_factory = lambda sr: T.MFCC(
        sample_rate=sr,
        n_mfcc=EMBEDDING_DIM,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH_FEATURE,
            "n_mels": N_MELS
        }
    ).to(device)
    amplitude_to_db_transform = T.AmplitudeToDB().to(device)

    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    if not audio_files:
        print(f"No audio files found in {audio_dir}.")
        return

    for audio_filename in tqdm(audio_files, desc="Processing audio files"):
        audio_path = os.path.join(audio_dir, audio_filename)

        try:
            audio_full, current_sr = torchaudio.load(audio_path)

            if audio_full.shape[0] > 1:
                audio_full = audio_full.mean(dim=0, keepdim=True)

            audio_full = audio_full.to(device)

            if current_sr != force_sr:
                print(f"Resampling {audio_filename} from {current_sr} Hz to {force_sr} Hz.")
                resampler = T.Resample(orig_freq=current_sr, new_freq=force_sr).to(device)
                audio_full = resampler(audio_full)
                current_sr = force_sr

            # Initialize MFCC transform
            mfcc_transform = mfcc_transform_factory(current_sr)

            # Calculate MFCC frame rate in Hz
            # This is based on the hop_length for the MFCC frames relative to the sample rate
            mfcc_frame_rate_hz = current_sr / HOP_LENGTH_FEATURE

            estimated_total_mfcc_frames = int(
                torch.ceil(torch.tensor(audio_full.shape[1] / HOP_LENGTH_FEATURE)).item()) + int(WINDOW_S * mfcc_frame_rate_hz / 2) # Add half window for buffer

            mfcc_frame_prob_accumulator = [[] for _ in range(estimated_total_mfcc_frames)]


            total_samples = audio_full.shape[1]
            window_samples = int(WINDOW_S * current_sr)
            hop_samples = int(HOP_S * current_sr)

            for start_raw_sample in range(0, total_samples, hop_samples):
                end_raw_sample = start_raw_sample + window_samples

                segment_audio = audio_full[:, start_raw_sample:end_raw_sample].to(device)

                if segment_audio.shape[1] < window_samples:
                    padding = window_samples - segment_audio.shape[1]
                    segment_audio = torch.nn.functional.pad(segment_audio, (0, padding))

                mfcc_features = mfcc_transform(segment_audio)
                mfcc_features = amplitude_to_db_transform(mfcc_features)

                with torch.no_grad():
                    segment_mfcc_frame_logits = model(mfcc_features)

                segment_mfcc_frame_probs = torch.softmax(segment_mfcc_frame_logits.squeeze(0), dim=1)

                start_mfcc_frame_for_segment_global = int(round(start_raw_sample / HOP_LENGTH_FEATURE))


                for i in range(segment_mfcc_frame_probs.shape[0]):
                    global_mfcc_frame_idx = start_mfcc_frame_for_segment_global + i
                    while global_mfcc_frame_idx >= len(mfcc_frame_prob_accumulator):
                        mfcc_frame_prob_accumulator.append([])

                    mfcc_frame_prob_accumulator[global_mfcc_frame_idx].append(segment_mfcc_frame_probs[i])

            averaged_mfcc_frame_probs_list = []
            for probs_list_for_frame in mfcc_frame_prob_accumulator:
                if probs_list_for_frame:
                    avg_prob = torch.stack(probs_list_for_frame).mean(dim=0)
                    averaged_mfcc_frame_probs_list.append(avg_prob)

            if not averaged_mfcc_frame_probs_list:
                print(f"Warning: No valid MFCC frames processed for {audio_filename}. Skipping output.")
                continue

            averaged_mfcc_frame_probs = torch.stack(averaged_mfcc_frame_probs_list).to(device)

            laughter_timestamps = postprocess(averaged_mfcc_frame_probs, mfcc_frame_rate_hz,
                                              thr=THRESHOLD, min_len=0.6, max_gap=0.4)

            output_data = {
                "filename": audio_filename,
                "laughter_timestamps": [[round(s, 3), round(e, 3)] for s, e in laughter_timestamps]
            }

            output_filepath = os.path.join(output_dir, f"{os.path.splitext(audio_filename)[0]}.json")
            with open(output_filepath, 'w') as f:
                json.dump(output_data, f, indent=2)

        except Exception as e:
            print(f"\nError processing {audio_filename}: {e}")

    print(f"\nInference complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()