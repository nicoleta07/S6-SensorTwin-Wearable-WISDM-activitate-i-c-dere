# S6-SensorTwin-Wearable-WISDM-activitate-i-c-dere
# S6 — SensorTwin Wearable (WISDM)

Proiect de **Human Activity Recognition (HAR)** folosind datele accelerometrului telefonului din subsetul `phone/accel` al datasetului WISDM.

## Obiectiv

Clasificarea activităților fizice înregistrate de accelerometrul telefonului, folosind ferestre de semnal de 10 secunde și caracteristici statistice.

> Important: implementarea actuală clasifică activitățile disponibile în WISDM. Nu este un detector de căderi.

## Date

- Dataset: [WISDM Smartphone and Smartwatch Activity and Biometrics Dataset — Kaggle](https://www.kaggle.com/datasets/nghhhhn/wisdm-dataset)
- Referință UCI: [DOI 10.24432/C5HK59](https://doi.org/10.24432/C5HK59)
- Senzor: phone accelerometer, axe X/Y/Z
- Frecvență nominală: 20 Hz
- Fereastră: 200 eșantioane (aprox. 10 secunde)
- Pas: 200 eșantioane (0% overlap)

Numărul de fișiere, participanți, înregistrări, clase și ferestre efectiv utilizate este calculat la rulare și salvat în CSV. Nu se confundă statisticile subsetului prelucrat cu statisticile datasetului WISDM integral.

## Reprezentare

Fiecare fereastră este descrisă prin 12 caracteristici:
- mean pentru X/Y/Z;
- standard deviation pentru X/Y/Z;
- minimum pentru X/Y/Z;
- maximum pentru X/Y/Z.

Alternativa metodologică este păstrarea ferestrei brute (200 × 3 valori) și folosirea unui model temporal, de exemplu CNN 1D. Această alternativă este o direcție viitoare și nu este implementată în versiunea curentă.

## Modele

### Model principal — MLP

- Input: 12 caracteristici
- Dense 128, ReLU
- Dropout 0.30
- Dense 64, ReLU
- Dropout 0.20
- Dense 32, ReLU
- Output: Softmax cu K clase

Antrenare: Adam, learning rate 0.001, sparse categorical cross-entropy, maximum 50 epochs, batch size 256, balanced class weights, EarlyStopping și ReduceLROnPlateau.

### Baseline — Random Forest

- 300 arbori
- `class_weight="balanced"`
- `random_state=42`

Ambele modele folosesc aceleași caracteristici și aceiași participanți în train și test. RF primește caracteristicile nenormalizate; MLP folosește StandardScaler ajustat numai pe train.

## Protocol experimental

Se utilizează `GroupShuffleSplit` cu participantul ca grup:
- aproximativ 64% train;
- aproximativ 16% validation;
- aproximativ 20% test.

Split-ul este subject-independent hold-out: același participant nu apare în mai multe subseturi. Nu este LOYO. Proporțiile reale pot diferi deoarece split-ul se face pe participanți, nu pe ferestre.

## Metrici

- Accuracy
- Macro F1-score
- Precision / Recall / F1 per clasă
- Confusion matrix
- Test loss pentru MLP

Valorile sunt completate numai după rularea codului.

## Structura repository-ului

```text
.
├── README.md
├── baseline_wisdm.py
├── datacard_S6.md
├── SOA_table.md
└── requirements.txt
```




Fișiere importante:
- `class_distribution.csv`
- `participant_distribution.csv`
- `split_overview.csv`
- `split_distribution.csv`
- `model_comparison.csv`
- `sensortwin_metrics.csv`
- `mlp_confusion_matrix.png`
- `rf_confusion_matrix.png`

## Referințe

- Weiss, G. (2019). *WISDM Smartphone and Smartwatch Activity and Biometrics Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59
- Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011). *Activity recognition using cell phone accelerometers*. ACM SIGKDD Explorations Newsletter, 12(2), 74–82. https://doi.org/10.1145/1964897.1964918

