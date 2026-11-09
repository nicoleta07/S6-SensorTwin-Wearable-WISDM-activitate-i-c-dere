# Data Card — S6: SensorTwin Wearable (WISDM)

## 1. Descrierea proiectului

SensorTwin Wearable este un proiect de recunoaștere a activităților umane (Human Activity Recognition — HAR) folosind date de mișcare colectate de telefon. Experimentul utilizează exclusiv semnalele accelerometrului telefonului din subsetul `phone/accel` al datasetului WISDM.

Scopul este clasificarea activităților prezente în date. Proiectul nu este, în forma actuală, un detector de căderi.

## 2. Sursa și versiunea datelor

- **Dataset:** WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.
- **Sursă utilizată în experiment:** copia WISDM găzduită pe Kaggle.
- **Link Kaggle:** https://www.kaggle.com/datasets/nghhhhn/wisdm-dataset
- **Referință UCI:** Weiss, G. (2019), DOI: https://doi.org/10.24432/C5HK59
- **Subset utilizat:** `phone/accel`.
- **Format așteptat:** câte un rând cu `subject, activity, timestamp, x, y, z`.

Documentația UCI descrie datasetul complet ca având 51 de subiecți, 18 activități și 15.630.426 instanțe, colectate cu senzori de telefon și ceas la 20 Hz. Acestea sunt statistici ale datasetului integral, nu trebuie confundate cu subsetul efectiv procesat de acest script. Numărul real de fișiere, înregistrări, participanți, clase și ferestre utilizate se obține la rularea scriptului și se raportează în fișierele CSV generate.

## 3. Senzorii utilizați

Sunt utilizate numai măsurătorile accelerometrului telefonului:
- axa X;
- axa Y;
- axa Z.

Nu sunt utilizate `phone/gyro`, `watch/accel` sau `watch/gyro`. Prin urmare, rezultatele se referă la reprezentarea construită din accelerometrul telefonului, nu la fuziunea mai multor senzori.

## 4. Structura datelor și preprocesare

Fiecare înregistrare este interpretată prin șase câmpuri:
- `subject` — identificatorul participantului;
- `activity` — eticheta activității;
- `timestamp` — marcaj temporal;
- `x`, `y`, `z` — accelerația pe cele trei axe.

Scriptul:
1. citește fișierele de date din `phone/accel`;
2. convertește timestamp-ul și valorile X/Y/Z la tip numeric;
3. elimină înregistrările cu valori numerice invalide;
4. elimină duplicatele pe baza participantului, activității, timestamp-ului și valorilor X/Y/Z;
5. sortează după participant și timestamp;
6. construiește ferestre separat pentru fiecare pereche participant–activitate, în interiorul fiecărui fișier.

Clasele care au mai puțin de două ferestre sunt eliminate înainte de split, pentru ca scriptul să poată încerca o împărțire train/validation/test. Clasele eliminate sunt afișate la rulare.

## 5. Segmentarea în ferestre

- **Dimensiunea ferestrei:** 200 eșantioane.
- **Frecvența nominală:** 20 Hz.
- **Durata nominală:** 10 secunde.
- **Pasul ferestrei:** 200 eșantioane.
- **Suprapunere:** 0%.

Durata de 10 secunde este calculată din frecvența nominală de 20 Hz. Ferestrele incomplete de la finalul fiecărui grup sunt ignorate.

## 6. Reprezentarea aleasă și alternativa

### Reprezentarea implementată

Fiecare fereastră este transformată într-un vector cu 12 caracteristici statistice:
- media X, Y, Z;
- deviația standard X, Y, Z;
- minimul X, Y, Z;
- maximul X, Y, Z.

Rezultă 4 caracteristici pentru fiecare dintre cele 3 axe, deci 12 valori de intrare per fereastră.

### Alternativă de comparat

O alternativă metodologică este reprezentarea brută a ferestrei, păstrând secvența temporală a celor trei axe (200 × 3 valori), fără agregarea în cele 12 statistici. Aceasta ar putea fi evaluată ulterior cu un model temporal, de exemplu un CNN 1D. Această alternativă este propusă, nu este implementată în experimentul curent.

## 7. Modelele

### Model principal: MLP

Rețeaua neuronală Multilayer Perceptron este implementată în TensorFlow/Keras:
- intrare: 12 caracteristici;
- Dense(128), activare ReLU;
- Dropout(0.30);
- Dense(64), activare ReLU;
- Dropout(0.20);
- Dense(32), activare ReLU;
- strat de ieșire Dense(K), Softmax, unde K este numărul claselor păstrate.

Caracteristicile sunt standardizate cu `StandardScaler`. Scaler-ul este ajustat exclusiv pe setul de train și apoi aplicat pe validation și test. Optimizarea folosește Adam cu rata inițială de învățare 0.001 și `sparse_categorical_crossentropy`. Se folosesc ponderi de clasă calculate numai din setul de train.

Antrenarea are maximum 50 de epoci și batch size 256. `EarlyStopping` monitorizează `val_loss`, cu patience 7 și restaurarea celor mai bune ponderi. `ReduceLROnPlateau` reduce rata de învățare când `val_loss` stagnează.

### Baseline: Random Forest

Random Forest este baseline-ul clasic de Machine Learning. Configurația din script este:
- 300 de arbori (`n_estimators=300`);
- `class_weight="balanced"`;
- `random_state=42`;
- `n_jobs=-1`;
- fără limită explicită pentru adâncimea arborilor.

Random Forest este antrenat pe aceleași caracteristici statistice și aceiași indici de train/test ca MLP. Nu are nevoie de standardizarea caracteristicilor; de aceea primește valorile brute ale celor 12 caracteristici. Compararea pe același test face evaluarea mai echitabilă.

## 8. Protocol experimental și justificarea split-ului

Se utilizează `GroupShuffleSplit`, având identificatorul participantului drept variabilă de grup. Niciun participant nu trebuie să apară în mai mult de un subset.

Protocolul urmărește aproximativ:
- **Train:** 64% din ferestre;
- **Validation:** 16% din ferestre;
- **Test:** 20% din ferestre.

Aceste procente sunt aproximative: împărțirea este realizată pe participanți, iar numărul de ferestre al fiecărui participant poate fi diferit. Împărțirea se face în doi pași: 20% dintre participanți/grupuri pentru test, apoi 20% din partea rămasă pentru validation. Scriptul încearcă mai multe split-uri pentru a găsi un set de train care conține toate clasele.

**De ce split pe participanți?** Ferestrele aceleiași persoane pot avea caracteristici similare. Dacă ferestre ale aceluiași participant ar ajunge simultan în train și test, evaluarea ar putea supraestima capacitatea de generalizare. Split-ul pe grupuri estimează mai bine performanța pe participanți nevăzuți la antrenare.

Acesta este un split subject-independent de tip hold-out. Nu este LOYO (Leave-One-Subject-Out) și nu este un split temporal în cadrul aceluiași participant.

## 9. Statistici efective ale datelor

Completați acest tabel după rularea scriptului. Valorile nu trebuie preluate din descrierea datasetului integral, ci din fișierele generate de experiment.

| Indicator | Valoare efectivă |
|---|---:|
| Fișiere candidate găsite | de completat din rulare |
| Fișiere citite cu succes | de completat din rulare |
| Înregistrări valide după curățare | de completat din rulare |
| Participanți reprezentați în ferestre | de completat din rulare |
| Clase păstrate | de completat din rulare |
| Ferestre totale | de completat din rulare |
| Caracteristici per fereastră | 12 |

### Distribuția pe activități

Se completează din `class_distribution.csv`, care conține:
- activitatea;
- numărul de ferestre;
- numărul de participanți care au acea activitate;
- procentul ferestrelor din total.

### Distribuția pe subseturi

Se completează din `split_overview.csv` și `split_distribution.csv`, care raportează numărul de ferestre și participanți pentru Train, Validation și Test, precum și distribuția claselor în fiecare subset.

## 10. Metrici de evaluare

Evaluarea finală se face pe subsetul de test, care conține participanți nefolosiți la antrenare sau validare.

Metrici raportate:
- **Accuracy:** proporția predicțiilor corecte;
- **Macro F1-score:** media F1 calculată separat pentru fiecare clasă, cu ponderi egale între clase;
- **Precision, Recall și F1 per clasă:** pentru analiza performanței pe fiecare activitate;
- **Confusion matrix:** pentru a observa confuziile dintre activități;
- **Test loss:** pentru MLP.

Rezultatele MLP și Random Forest sunt salvate în `model_comparison.csv` și `sensortwin_metrics.csv`. Nu introduceți valori în acest Data Card înainte de rularea efectivă.

## 11. Fișiere generate

În directorul de output (implicit `/kaggle/working/`) sunt salvate:
- `class_distribution.csv` — ferestre și participanți per activitate;
- `participant_distribution.csv` — ferestre per participant;
- `split_distribution.csv` — ferestre și participanți per clasă și subset;
- `split_overview.csv` — total ferestre și participanți per subset;
- `model_comparison.csv` — Accuracy și Macro F1 pentru MLP și Random Forest;
- `sensortwin_metrics.csv` — rezumatul experimentului;
- `mlp_classification_report.csv` și `rf_classification_report.csv`;
- `class_distribution.png`;
- `mlp_accuracy.png`, `mlp_loss.png`;
- `mlp_confusion_matrix.png`, `rf_confusion_matrix.png`;
- `sensortwin_mlp.keras`;
- `sensortwin_random_forest.joblib`;
- `sensortwin_scaler.joblib`;
- `sensortwin_label_encoder.joblib`.

## 12. Limitări și utilizare responsabilă

- Experimentul folosește doar accelerometrul telefonului.
- Caracteristicile statistice rezumă fiecare fereastră și pierd informație temporală fină.
- Ferestrele incomplete sunt ignorate.
- Split-ul este subject-independent hold-out, nu LOYO.
- Rezultatele pot depinde de distribuția participanților și a activităților.
- WISDM, în configurația folosită aici, este utilizat pentru recunoașterea activităților etichetate. În absența unei etichete explicite de cădere, modelul nu trebuie descris ca detector de căderi.
- Proiectul are scop educațional și experimental și nu este un dispozitiv medical sau un sistem de siguranță validat.

## 13. Referință pentru dataset

Weiss, G. (2019). *WISDM Smartphone and Smartwatch Activity and Biometrics Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HK59

Datasetul original WISDM și informațiile despre el sunt descrise de asemenea pe pagina UCI: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
