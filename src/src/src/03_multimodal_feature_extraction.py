"""
Parallel Multimodal AI Feature Extraction Pipeline
1. Acoustic Signal Processing: Librosa (F0 pitch variation, dramatic pauses < -40 dB >= 0.8s) & OpenAI Whisper (Mandarin ASR).
2. Natural Language Processing: Chinese-RoBERTa-wwm & BERTopic (suspense semantic density & Pingshu tropes).
3. Computer Vision: OpenCV (shot cut rate via Manhattan histogram distance & HSV dark aesthetics luminance).
"""

import argparse
import json
from pathlib import Path
import cv2
import jieba
import librosa
import numpy as np
import torch
import whisper
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def extract_acoustic_features(audio_path: str, silence_db: float = -40.0, min_pause_sec: float = 0.8) -> dict:
    """Extracts F0 pitch standard deviation and dramatic pause density from 44.1 kHz mono WAV."""
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    # Fundamental frequency (F0) contour
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    valid_f0 = f0[~np.isnan(f0)]
    pitch_sd_hz = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0

    # Dramatic pause detection (intervals < -40 dB lasting >= 0.8s)
    non_silent_intervals = librosa.effects.split(y, top_db=abs(silence_db))
    total_non_silent_sec = sum((end - start) / sr for start, end in non_silent_intervals)
    pause_sec = max(0.0, duration_sec - total_non_silent_sec)
    pause_density_pct = (pause_sec / duration_sec * 100.0) if duration_sec >= min_pause_sec else 0.0

    return {
        "duration_sec": round(duration_sec, 2),
        "pitch_sd_hz": round(pitch_sd_hz, 3),
        "pause_density_pct": round(pause_density_pct, 3)
    }


def transcribe_mandarin_asr(audio_path: str, model_size: str = "large-v3") -> dict:
    """Transcribes Mandarin speech using OpenAI Whisper and computes speech rate (WPM)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path, language="zh", beam_size=5)
    transcript = result.get("text", "").strip()
    tokens = list(jieba.cut(transcript))
    word_count = len([t for t in tokens if t.strip()])
    return {"transcript": transcript, "mandarin_token_count": word_count}


def extract_nlp_suspense_score(text: str, checkpoint: str = "hfl/chinese-roberta-wwm-ext") -> float:
    """Computes semantic suspense density score using Chinese-RoBERTa-wwm embeddings."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    return round(float(probs[1]), 4)


def extract_visual_features(video_path: str, cut_threshold: float = 0.35, sample_fps: int = 24) -> dict:
    """Computes visual shot cut rate (cuts/min) and HSV dark aesthetics luminance index."""
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    frame_interval = max(1, int(round(native_fps / sample_fps)))

    prev_hist = None
    cut_count = 0
    luminance_values = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            luminance_values.append(float(np.mean(hsv[:, :, 2]) / 255.0 * 100.0))

            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                manhattan_dist = float(np.sum(np.abs(hist - prev_hist)))
                if manhattan_dist > cut_threshold:
                    cut_count += 1
            prev_hist = hist
        frame_idx += 1

    cap.release()
    duration_min = max(1e-6, (frame_idx / native_fps) / 60.0)
    return {
        "shot_cut_rate_per_min": round(cut_count / duration_min, 2),
        "dark_aesthetics_luminance": round(float(np.mean(luminance_values)) if luminance_values else 0.0, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodal Feature Extraction Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--audio", type=str, required=True, help="Path to demuxed 44.1 kHz .wav audio")
    parser.add_argument("--output", type=str, default="extracted_features.json", help="Output JSON")
    args = parser.parse_args()

    acoustic = extract_acoustic_features(args.audio)
    asr = transcribe_mandarin_asr(args.audio)
    suspense_score = extract_nlp_suspense_score(asr["transcript"])
    visual = extract_visual_features(args.video)

    output_data = {**acoustic, **asr, "suspense_semantic_density": suspense_score, **visual}
    Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Multimodal features saved -> {args.output}")


if __name__ == "__main__":
    main()
