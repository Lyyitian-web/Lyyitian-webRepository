# AI-Enabled Cultural-Psychological Framework of Parasocial Interaction, Flow, and User Stickiness in Short-Form Suspense Audiobooks on Douyin

This repository contains the official dataset and analytical codebase for the study: **"An AI-Enabled Cultural-Psychological Framework of Parasocial Interaction, Flow, and User Stickiness in Short-Form Suspense Audiobooks on Douyin"**.

## 📌 Repository Overview

This project bridges **Social Data Science**, **Human-Computer Interaction (HCI)**, and **Media Psychology**. It implements an auditable, non-invasive computational framework to extract multimodal features from short-form digital media (Douyin/TikTok super-app ecosystem) and model their impact on audience retention (user stickiness) and algorithmic circulation velocity ($V_{circ}$).

---

## 📁 Repository Structure

```text
.
├── README.md                                   # Repository documentation
├── LICENSE                                     # Open-source license (MIT)
├── requirements.txt                            # Python dependencies
├── data/
│   ├── douyin_suspense_audiobook_audited_dataset.csv   # Audited empirical dataset (N=4,218)
│   └── codebook.md                             # Detailed feature & column definitions
├── src/
│   ├── scraper/
│   │   └── playwright_extractor.py             # Headless browser web extraction module
│   ├── data_audit/
│   │   ├── benford_audit.py                    # Benford's Law 1st digit distribution audit
│   │   └── zscore_purger.py                    # Z-score anomaly filtering for bot detection
│   ├── ai_pipeline/
│   │   ├── acoustic_librosa_whisper.py         # Acoustic feature extraction (F0, silences, WPM)
│   │   ├── nlp_roberta_bertopic.py             # Chinese-RoBERTa-wwm & BERTopic trope classifier
│   │   └── vision_opencv.py                    # OpenCV shot cut rate & dark aesthetics index
│   └── modeling/
│       ├── pls_sem_model.R                     # PLS-SEM structural equation modeling script
│       └── random_forest_importance.py         # Supervised Random Forest feature ranking
```

---

## 📊 Dataset Description (`data/douyin_suspense_audiobook_audited_dataset.csv`)

The empirical dataset consists of **N=4,218 validated records** (purged from N=4,500 initial records via Benford's Law and Z-score anomaly filtering).

### Feature Dictionary

| Column Name | Category | Type / Unit | Description |
| :--- | :--- | :--- | :--- |
| `episode_id` | Identifier | String | Unique audiobook video episode ID |
| `account_id` | Identifier | String | Anonymized creator account ID |
| `platform` | Metadata | String | Social platform source (Douyin) |
| `duration_sec` | Metadata | Continuous (s) | Total audio-video duration in seconds |
| `speech_rate_wpm` | Acoustic AI | Continuous (WPM) | Words per minute transcribed via OpenAI Whisper AI (`whisper-large-v3`) |
| `pitch_var_hz` | Acoustic AI | Continuous (Hz) | Standard deviation of fundamental pitch $\sigma_{F0}$ via Librosa |
| `pause_density_pct` | Acoustic AI | Continuous (%) | Proportion of silent pauses ($\ge 0.8	ext{ s}$, intensity $< -40	ext{ dB}$) |
| `suspense_semantic_score` | Text / NLP | Continuous (0–1) | Suspense probability score via Chinese-RoBERTa-wwm (`hfl/chinese-roberta-wwm-ext`) |
| `pingshu_cultural_index` | Cultural NLP | Continuous (0–1) | Frequency of Pingshu oral storytelling syntaxes (BERTopic v0.16.0) |
| `shot_cut_rate_pm` | Vision AI | Continuous (cuts/min) | Visual shot cut rate computed via OpenCV keyframe histogram distance |
| `dark_aesthetics_index` | Vision AI | Continuous (0–100) | Mean HSV pixel luminance (dark visual background alignment) |
| `ocr_text_pct` | Vision AI | Continuous (%) | Percentage of visual area covered by text overlays |
| `ecommerce_tag_present` | Interface | Binary (0/1) | Presence of direct commercial call-to-action e-commerce widgets |
| `benford_audit_pass` | Data Quality | Binary (1) | Pass flag for Benford's Logarithmic Law audit ($\chi^2$ test, $p \ge 0.01$) |
| `zscore_anomaly_val` | Data Quality | Continuous | Interaction anomaly score ($Z_i \le 3.5$) |
| `psi_score` | Psychometric | Latent (1–5) | Parasocial Interaction score with the narrator's voice |
| `flow_immersion_score` | Psychometric | Latent (1–5) | Cognitive Flow state & temporal distortion score |
| `likes_t0` | Engagement | Integer | Absolute count of likes at initial observation $T_0$ |
| `comments_t0` | Engagement | Integer | Absolute count of comments at $T_0$ |
| `shares_t0` | Engagement | Integer | Absolute count of shares at $T_0$ |
| `collections_t0` | Engagement | Integer | Absolute count of collections/bookmarks at $T_0$ |
| `engagement_index_ei` | Dependent | Continuous | Global Engagement Index $E_i = 1.0	ext{Likes} + 2.5	ext{Comments} + 4.0	ext{Shares} + 3.5	ext{Collections}$ |
| `log_engagement_yi` | Dependent | Continuous | Skewness-stabilized engagement $y_i = \ln(E_i + 1)$ |
| `circ_velocity_vcirc_pct` | Dependent | Continuous (%/7d) | Algorithmic circulation velocity percentage growth between $T_0$ and $T_1 = T_0 + 7	ext{ days}$ |

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/anonymous-repo/douyin-suspense-audiobook-dataset.git
cd douyin-suspense-audiobook-dataset
pip install -r requirements.txt
```

### 2. Reproduce Machine Learning Feature Importance (Random Forest)

To train the Random Forest Regressor ($R^2 = 0.635$) and generate the feature importance distribution:

```python
from src.modeling.random_forest_importance import run_rf_analysis

# Runs 10-fold cross-validation with 1,000 decision trees
run_rf_analysis(data_path="data/douyin_suspense_audiobook_audited_dataset.csv")
```

### 3. Run Data Integrity Audit (Benford's Law & Z-Score)

To re-verify the bot-purging protocol on raw metadata:

```bash
python src/data_audit/benford_audit.py --input raw_data.csv --alpha 0.01
python src/data_audit/zscore_purger.py --input raw_data.csv --threshold 3.5
```

---

## 📜 Ethical Considerations & Data Privacy

This research strictly follows non-invasive Open-Source Intelligence (OSINT) principles:
- **No Private Data Collected:** Only publicly visible surface metadata, captions, and demuxed media tracks were audited.
- **No Personal Identifiable Information (PII):** All creator identifiers (`account_id`) and episode keys (`episode_id`) have been anonymized.
- **Terms of Service Compliance:** Web interactions were executed via headless browser session emulation without bypassing user access controls or private authentication walls.

---

## ⚖️ Citation & License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

If you use this dataset or codebase in your academic work, please cite:

```bibtex
@article{douyin_suspense_audiobooks_2024,
  title={An AI-Enabled Cultural-Psychological Framework of Parasocial Interaction, Flow, and User Stickiness in Short-Form Suspense Audiobooks on Douyin},
  journal={PeerJ Computer Science},
  year={2024},
  publisher={PeerJ Inc.}
}
```
