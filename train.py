# laughdetector/train.py
import datetime as dt, os, click, torch
from torch import nn, optim
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, random_split # Import random_split
from torchmetrics import Accuracy, Precision, Recall, F1Score

from laughdetector.load import load
from laughdetector.nn.tagger import LSTMTagger


@click.command()
@click.option("--audio-dir", required=True, help="Directory containing audio files.")
@click.option("--train-path", required=True, help="Path to the JSON annotation file.")
@click.option("--num-epochs", type=int, default=30, help="Number of training epochs.")
@click.option("--save-dir", type=str, default="./models", help="Directory to save the trained model.")
@click.option("--batch-size", type=int, default=1, help="Batch size for training (Start with 1 for 30s segments).")
@click.option("--num-workers", type=int, default=4, help="Number of worker processes for data loading.")
@click.option("--window-s", type=float, default=30.0, help="Window size in seconds for audio segments (default 30s).")
@click.option("--hop-s", type=float, default=1.0, help="Hop size in seconds for sliding window.")
@click.option("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
@click.option("--hidden-dim", type=int, default=64, help="Hidden dimension for the LSTM.")
@click.option("--n-mfcc", type=int, default=40, help="Number of MFCCs to extract.")
@click.option("--n-fft", type=int, default=2048, help="FFT window size for spectrogram/MFCCs.")
@click.option("--hop-length-feature", type=int, default=512, help="Hop length for MFCC frames.")
def main(audio_dir, train_path, num_epochs, save_dir, batch_size, num_workers, window_s, hop_s,
         lr, hidden_dim, n_mfcc, n_fft, hop_length_feature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing audio dataset (with MFCC features)...")
    full_dataset = load(audio_dir, train_path, window_s=window_s, hop_s=hop_s,
                        n_mfcc=n_mfcc, n_fft=n_fft, hop_length_feature=hop_length_feature)

    if len(full_dataset) == 0:
        print("No segments were loaded into the dataset. Training cannot proceed.")
        return

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset prepared with {len(train_dataset)} training and {len(val_dataset)} validation segments.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    sample_segment, _ = train_dataset[0]
    embedding_dim = sample_segment.size(0)
    print(f"Determined embedding_dim (number of features per frame): {embedding_dim}")

    model = LSTMTagger(embedding_dim, hidden_dim=hidden_dim, tagset_size=2).to(device)

    print("Calculating class weights for CrossEntropyLoss...")
    class_counts = torch.zeros(2, dtype=torch.long) # [laugh_count, non_laugh_count]
    for _, targets_batch in tqdm(train_loader, desc="Counting class occurrences"):
        flat_targets = targets_batch.view(-1)
        counts = torch.bincount(flat_targets, minlength=2)
        class_counts += counts.cpu()

    total_samples = class_counts.sum().item()
    if total_samples == 0:
        print("No samples found in training data for class weight calculation. Exiting.")
        return

    class_weights = total_samples / (2.0 * class_counts.float())
    class_weights = class_weights / class_weights.sum() * 2.0 # Normalize to sum to 2
    class_weights = class_weights.to(device)
    print(f"Calculated class weights: {class_weights.tolist()} (laugh, non-laugh)")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr) # Using Adam optimizer

    # Initialize evaluation metrics
    metrics = {
        'accuracy': Accuracy(task="multiclass", num_classes=2).to(device),
        'precision': Precision(task="multiclass", num_classes=2, average='macro').to(device),
        'recall': Recall(task="multiclass", num_classes=2, average='macro').to(device),
        'f1_score': F1Score(task="multiclass", num_classes=2, average='macro').to(device)
    }

    best_val_f1 = -1.0 # Initialize with a low value for tracking best model
    best_epoch = -1

    # Training Loop
    print("\nStarting training...")
    for epoch in trange(num_epochs, desc="Epoch"):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for batch_idx, (features_batch, targets_batch) in enumerate(tqdm(train_loader, desc=f"Train Batch (Epoch {epoch + 1})")):
            optimizer.zero_grad(set_to_none=True)

            features_batch = features_batch.to(device).contiguous()
            targets_batch = targets_batch.to(device)

            logits = model(features_batch)

            loss = loss_fn(logits.view(-1, logits.size(-1)), targets_batch.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1).view(-1)
            all_preds.append(preds)
            all_targets.append(targets_batch.view(-1))

        avg_train_loss = total_loss / len(train_loader)
        train_all_preds_tensor = torch.cat(all_preds)
        train_all_targets_tensor = torch.cat(all_targets)

        train_accuracy = metrics['accuracy'](train_all_preds_tensor, train_all_targets_tensor)
        train_precision = metrics['precision'](train_all_preds_tensor, train_all_targets_tensor)
        train_recall = metrics['recall'](train_all_preds_tensor, train_all_targets_tensor)
        train_f1 = metrics['f1_score'](train_all_preds_tensor, train_all_targets_tensor)


        # Validation Loop
        model.eval()
        val_total_loss = 0
        val_all_preds = []
        val_all_targets = []

        with torch.no_grad():
            for batch_idx, (features_batch, targets_batch) in enumerate(tqdm(val_loader, desc=f"Val Batch (Epoch {epoch + 1})")):
                features_batch = features_batch.to(device).contiguous()
                targets_batch = targets_batch.to(device)

                logits = model(features_batch)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets_batch.view(-1))
                val_total_loss += loss.item()

                preds = logits.argmax(dim=-1).view(-1)
                val_all_preds.append(preds)
                val_all_targets.append(targets_batch.view(-1))

        avg_val_loss = val_total_loss / len(val_loader)
        val_all_preds_tensor = torch.cat(val_all_preds)
        val_all_targets_tensor = torch.cat(val_all_targets)

        val_accuracy = metrics['accuracy'](val_all_preds_tensor, val_all_targets_tensor)
        val_precision = metrics['precision'](val_all_preds_tensor, val_all_targets_tensor)
        val_recall = metrics['recall'](val_all_preds_tensor, val_all_targets_tensor)
        val_f1 = metrics['f1_score'](val_all_preds_tensor, val_all_targets_tensor)


        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | "
            f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1-Score: {val_f1:.4f}"
        )

        for metric in metrics.values():
            metric.reset()

        # Save the best model based on validation F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"best_model_epoch_{best_epoch:03d}_val_f1_{best_val_f1:.4f}.bin")
            torch.save(model.state_dict(), out_path) # Save state_dict, not the whole model
            print(f"--> Saved best model to {out_path}")

    print(f"\nTraining finished. Best Validation F1-Score: {best_val_f1:.4f} at Epoch {best_epoch}")
    print(f"Final model saved as best_model_epoch_{best_epoch:03d}_val_f1_{best_val_f1:.4f}.bin")


if __name__ == "__main__":
    main()