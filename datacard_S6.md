# Data Card — S6: SensorTwin Wearable (WISDM)

## 1. Descrierea proiectului

SensorTwin Wearable este un proiect de Machine Learning dedicat recunoașterii activităților fizice pe baza datelor colectate de senzori de mișcare. Scopul experimentului este de a antrena și evalua o rețea neuronală care poate clasifica activități folosind semnalele de accelerație ale telefonului.

Pentru acest experiment am folosit datasetul WISDM și am construit un flux de lucru care include preprocesarea datelor, extragerea caracteristicilor, antrenarea pe epoci și evaluarea pe un set de testare separat.

## 2. Sursa datelor

- **Dataset:** WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
- **Platformă utilizată:** Kaggle
- **Link:** https://www.kaggle.com/datasets/nghhhhn/wisdm-dataset
- **Calea din notebook:** `/kaggle/input/datasets/nghhhhn/wisdm-dataset`

Datasetul original WISDM conține date de la 51 de participanți care au realizat 18 activități și aproximativ 15,6 milioane de înregistrări, colectate la o frecvență de 20 Hz. Aceste valori descriu datasetul original; numărul efectiv de exemple folosite în experiment este afișat la rularea codului.

## 3. Senzorii utilizați

În acest experiment am folosit **exclusiv fișierele din folderul `phone/accel`**, care conține datele de accelerometru ale telefonului.

Nu am utilizat datele din `phone/gyro`, `watch/accel` sau `watch/gyro`. Astfel, modelul se bazează doar pe accelerația telefonului pe axele X, Y și Z.

Codul caută toate fișierele de date compatibile din folderul selectat, fără o limită prestabilită a numărului de fișiere sau a numărului de rânduri per fișier.

## 4. Structura datelor

Fișierele sunt interpretate în formatul cu șase coloane:

- `subject` — identificatorul participantului;
- `activity` — activitatea realizată;
- `timestamp` — momentul înregistrării;
- `x`, `y`, `z` — valorile accelerației pe cele trei axe.

În etapa de preprocesare sunt eliminate valorile numerice invalide și înregistrările duplicate din fiecare fișier. Datele sunt apoi ordonate temporal și grupate după participant și activitate.

## 5. Extragerea caracteristicilor

Semnalele sunt împărțite în ferestre de 200 de eșantioane, fără suprapunere. La frecvența de 20 Hz, o fereastră reprezintă aproximativ 10 secunde.

Pentru fiecare fereastră sunt calculate 12 caracteristici statistice:
- media pentru axele X, Y și Z;
- deviația standard pentru fiecare axă;
- valoarea minimă pentru fiecare axă;
- valoarea maximă pentru fiecare axă.

Aceste caracteristici transformă semnalele brute într-o reprezentare numerică pe care modelul o poate folosi pentru clasificare.

## 6. Modelul de Machine Learning

Pentru experiment am ales o rețea neuronală de tip **Multilayer Perceptron (MLP)**, construită cu TensorFlow/Keras.

Arhitectura conține trei straturi dense ascunse, cu 128, 64 și 32 de neuroni. Straturile folosesc funcția de activare ReLU, iar două straturi Dropout sunt utilizate pentru a reduce riscul de supraînvățare. Stratul final folosește Softmax pentru a produce scoruri pentru activitățile clasificate.

Înainte de antrenare, caracteristicile sunt standardizate folosind un `StandardScaler` antrenat exclusiv pe setul de antrenare. Modelul este optimizat cu Adam și funcția de pierdere `sparse_categorical_crossentropy`. Se folosesc ponderi pentru clase pentru a reduce efectul dezechilibrului dintre numărul de exemple.

## 7. Antrenarea și validarea

Modelul este configurat pentru maximum 50 de epoci, cu un batch size de 256. La fiecare epocă sunt urmărite performanțele pe setul de antrenare și pe setul de validare.

Pentru controlul antrenării sunt folosite:
- **EarlyStopping:** oprește antrenarea dacă pierderea pe validare nu se mai îmbunătățește și restaurează cele mai bune ponderi;
- **ReduceLROnPlateau:** reduce rata de învățare când performanța pe validare stagnează.

Prin urmare, modelul poate termina antrenarea înainte de 50 de epoci dacă nu mai observă îmbunătățiri. Numărul efectiv de epoci este afișat la finalul rulării.

## 8. Împărțirea datelor

Codul încearcă să împartă datele în trei seturi, pe baza identificatorilor participanților:
- aproximativ 64% pentru antrenare;
- aproximativ 16% pentru validare;
- aproximativ 20% pentru testare.

Proporțiile sunt aproximative și se referă la ferestrele rezultate din împărțirea grupurilor de participanți. Codul caută o împărțire în care toate clasele să fie prezente în setul de antrenare.

Setul de validare este utilizat pentru monitorizarea antrenării și alegerea celor mai bune ponderi. Setul de testare este păstrat separat pentru evaluarea finală.

## 9. Evaluarea performanței

Modelul este evaluat pe setul de testare folosind următoarele metrici:

- **Accuracy:** proporția predicțiilor corecte;
- **Macro F1-score:** media scorurilor F1 pentru clase, acordând aceeași importanță fiecărei clase;
- **Test loss:** valoarea funcției de pierdere pe setul de testare.

Notebook-ul afișează și un raport pe clase, o matrice de confuzie și graficele acurateței și pierderii pe epoci. Acestea permit observarea performanței modelului și a eventualelor semne de supraînvățare.

Valorile finale se completează pe baza rulării efective a notebook-ului, nu sunt presupuse în avans.

## 10. Fișiere generate

La final, codul salvează:
- `sensortwin_metrics.csv` — rezumatul metricilor și al dimensiunii seturilor;
- `sensortwin_mlp.keras` — modelul neuronal antrenat.

Ambele fișiere sunt salvate în `/kaggle/working/`.

## 11. Limitări și utilizare responsabilă

Experimentul folosește numai accelerometrul telefonului și nu include informațiile suplimentare care ar putea fi oferite de giroscop sau de ceas.

WISDM este destinat recunoașterii activităților fizice și nu conține, în versiunea standard, o clasă dedicată căderilor. Din acest motiv, modelul nu poate fi prezentat ca un detector de căderi. Pentru o astfel de funcționalitate este necesar un dataset cu exemple etichetate explicit ca „fall” și activități normale, precum și o evaluare specifică acestei sarcini.

Rezultatele depind de calitatea datelor, de activitățile disponibile și de participanții incluși. Acest proiect are un scop educațional și experimental și nu reprezintă un sistem medical sau de siguranță validat.

## 12. Concluzie

În cadrul proiectului SensorTwin Wearable am construit o rețea neuronală MLP pentru recunoașterea activităților folosind datele de accelerometru ale telefonului din WISDM. Am extras caracteristici statistice din ferestre de semnal, am standardizat datele, am antrenat modelul pe epoci și am evaluat performanța folosind un set de testare separat.

Această abordare permite urmărirea procesului de învățare și oferă o bază pentru experimente viitoare, inclusiv compararea cu alte arhitecturi și dezvoltarea unui sistem separat de detectare a căderilor.
