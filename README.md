# An AI-Enabled Cultural-Psychological Framework of Parasocial Interaction, Flow, and User Stickiness in Short-Form Suspense Audiobooks on Douyin

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24199332.svg)](https://doi.org/10.5281/zenodo.24199332)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 1. Title
**An AI-Enabled Cultural-Psychological Framework of Parasocial Interaction, Flow, and User Stickiness in Short-Form Suspense Audiobooks on Douyin**

## 2. Description
This repository provides the reproducible Python computational pipeline and anonymized dataset ($N = 4,218$ validated records) accompanying the *PeerJ Computer Science* (AI Application) manuscript. The framework integrates non-invasive OSINT web extraction (Playwright), two-stage statistical bot auditing (Benford's Law and $Z$-score filtering), multimodal AI feature extraction (Librosa, OpenAI Whisper AI, Chinese-RoBERTa-wwm, BERTopic, and OpenCV), Partial Least Squares Structural Equation Modeling (PLS-SEM), and supervised machine learning (Random Forest Regressor) to predict audience retention and content circulation velocity ($V_{circ}$).

## 3. Dataset Information
* **Source & Scope:** Publicly accessible short-form suspense audiobooks and traditional mystery narratives (*Pingshu* / *Guochao* genres) collected from Douyin over a 90-day longitudinal window ($T_0$ and $T_1 = T_0 + 7$ days).
* **Sample Size:** $N = 4,500$ raw instances collected; $282$ non-organic/bot records ($6.26\%$) purged via Benford's Law ($\chi^2, p < 0.01$) and $Z$-score ($> 3.5$) auditing, yielding **$N = 4,218$ validated records**.
* **Privacy & Anonymization:** All creator handles, user identifiers, and video URLs have been anonymized (`VID_XXX`, `creator_XXX`) in compliance with OSINT ethical guidelines and data protection standards.
* **Key Variables:**
  * `Vcirc`: Circulation velocity (% change in engagement between $T_0$ and $T_1$).
  * `Si`: Composite User Stickiness index.
  * `pitch_sd_hz` ($\sigma_{F0}$): Fundamental frequency pitch standard deviation (Hz).
  * `pause_density_pct`: Proportion of audio duration occupied by dramatic silent pauses ($\ge 0.8\text{ s}$, $<-40\text{ dB}$).
  * `speech_rate_wpm`: Mandarin speech rate (words per minute).
  * `suspense_density`: Semantic suspense probability score ($0\text{--}1$).
  * `pingshu_index`: Traditional oral storytelling cultural trope score ($0\text{--}1$).
  * `shot_cut_rate`: Visual transition frequency (cuts/min).
  * `dark_aesthetics_index`: Mean HSV pixel luminance score ($0\text{--}100$).
  * `psi_score` & `flow_score`: Latent psychometric scores for Parasocial Interaction and Flow State ($1\text{--}5$ scale).

## 4. Code Information
* `src/01_playwright_extractor.py`: Headless browser DOM/JSON collector for public metadata and longitudinal $T_0/T_1$ snapshots.
* `src/02_statistical_audit.py`: First-digit Benford's Law $\chi^2$ goodness-of-fit test and $Z$-score ($>3.5$) anomaly purging module.
* `src/03_multimodal_feature_extraction.py`: Parallel feature extraction using FFmpeg + Librosa ($F_0$ pitch and dramatic pauses), `whisper-large-v3` (Mandarin ASR), `hfl/chinese-roberta-wwm-ext` + `BERTopic` (suspense semantics and *Pingshu* tropes), and OpenCV (shot cuts and HSV luminance).
* `src/04_pls_sem_and_random_forest.py`: Structural path estimation, moderation analysis (`Flow x Pingshu` and `OCR x E-commerce`), and 10-fold cross-validated Random Forest Regressor (`n_estimators=1000`, `max_depth=15`).

## 5. Usage Instructions
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Lyyitian-web/Lyyitian-webRepository.git](https://github.com/Lyyitian-web/Lyyitian-webRepository.git)
   cd Lyyitian-webRepository
