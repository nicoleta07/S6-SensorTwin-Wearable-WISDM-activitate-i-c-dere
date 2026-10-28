# Data Card — S6: SensorTwin Wearable (WISDM)

## 1. Descrierea proiectului

SensorTwin Wearable este un proiect de Machine Learning care urmărește recunoașterea activităților fizice folosind date colectate de senzori de mișcare. Scopul este de a antrena un model care să identifice activitatea realizată pe baza mișcărilor înregistrate de un dispozitiv mobil sau purtabil.

Proiectul folosește datasetul WISDM și algoritmul Random Forest pentru clasificarea activităților. Acesta reprezintă o etapă inițială în dezvoltarea unui sistem de monitorizare a mișcării și poate constitui o bază pentru cercetări ulterioare privind detectarea căderilor.

## 2. Sursa și dimensiunea datasetului

- **Numele datasetului:** WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
- **Sursa folosită în proiect:** Kaggle
- **Link:** https://www.kaggle.com/datasets/nghhhhn/wisdm-dataset
- **Sursa originală:** UCI Machine Learning Repository
- **Numărul de participanți în datasetul original:** 51
- **Numărul de activități în datasetul original:** 18
- **Numărul aproximativ de înregistrări în datasetul original:** 15,6 milioane
- **Frecvența de eșantionare:** 20 Hz

Aceste valori descriu datasetul original WISDM. Numărul efectiv de participanți, înregistrări și activități citite de cod poate fi diferit, în funcție de fișierele incluse în copia Kaggle și de datele care trec de etapa de preprocesare.

## 3. Structura datelor

Codul este conceput să citească fișiere cu șase coloane:

- `subject` — identificatorul participantului;
- `activity` — activitatea efectuată;
- `timestamp` — momentul înregistrării;
- `x` — accelerația pe axa X;
- `y` — accelerația pe axa Y;
- `z` — accelerația pe axa Z.

Valorile de pe cele trei axe descriu mișcarea dispozitivului în timpul activităților. Codul selectează fișierele care par să conțină date de accelerometru, dacă denumirile acestora permit identificarea lor.

## 4. Preprocesarea datelor

Înainte de antrenarea modelului, programul caută fișierele disponibile în Kaggle Input și încearcă să dezarhiveze automat fișierele ZIP.

Datele sunt apoi citite, valorile invalide sunt eliminate, iar înregistrările duplicate sunt îndepărtate. Semnalele sunt sortate după fișier, participant și timestamp.

Ulterior, datele sunt împărțite în ferestre de 200 de eșantioane. La o frecvență de 20 Hz, o fereastră reprezintă aproximativ 10 secunde de mișcare. Ferestrele nu se suprapun.

Pentru fiecare fereastră sunt calculate 12 caracteristici statistice: media, deviația standard, valoarea minimă și valoarea maximă pentru fiecare dintre axele X, Y și Z.

Ferestrele care aparțin unor clase cu mai puțin de două exemple sunt eliminate înainte de antrenare.

## 5. Modelul utilizat

Pentru clasificarea activităților este folosit algoritmul **Random Forest**, configurat cu 100 de arbori de decizie.

Modelul învață relația dintre caracteristicile extrase din semnalele senzorilor și etichetele activităților. Pentru a ține cont de eventualele diferențe de reprezentare dintre clase, este folosită și opțiunea `class_weight="balanced"`.

## 6. Împărțirea datelor

Codul încearcă să folosească aproximativ 80% dintre grupurile de participanți pentru antrenare și 20% pentru testare. Împărțirea se face prin `GroupShuffleSplit`, cu `random_state=42`, astfel încât datele aceluiași participant să nu ajungă atât în antrenare, cât și în testare.

Proporția se referă la grupurile de participanți, nu garantează exact 80% dintre înregistrări. Dacă există un singur participant, codul folosește o împărțire aleatorie a ferestrelor, ceea ce oferă o evaluare mai puțin concludentă pentru persoane noi.

## 7. Evaluarea rezultatelor

Performanța modelului este evaluată cu două metrici:

- **Accuracy:** procentul predicțiilor corecte pe setul de testare.
- **Macro F1-score:** media scorurilor F1 pentru clase, acordând fiecărei clase aceeași importanță.

La rularea notebook-ului sunt afișate acuratețea, Macro F1-score, numărul total de ferestre, numărul ferestrelor de antrenare și testare și numărul activităților clasificate.

Rezultatele efective trebuie completate după executarea codului. Acestea nu pot fi stabilite doar din configurația modelului.

## 8. Fișierul cu rezultate

Programul salvează un fișier numit `sensortwin_metrics.csv` în directorul `/kaggle/working/`.

Acesta conține un rezumat al performanței modelului și al numărului de ferestre folosite. Nu sunt exportate toate înregistrările senzorilor.



## 9. Concluzie

Proiectul SensorTwin Wearable demonstrează un flux de lucru de Machine Learning pentru recunoașterea activităților pe baza datelor de accelerometru. Acesta include citirea și curățarea datelor, extragerea caracteristicilor, antrenarea unui model Random Forest și evaluarea prin Accuracy și Macro F1-score.

Rezultatele obținute vor arăta cât de bine recunoaște modelul activitățile din setul de testare și vor oferi un punct de plecare pentru dezvoltarea ulterioară a unui sistem de monitorizare a mișcării.
