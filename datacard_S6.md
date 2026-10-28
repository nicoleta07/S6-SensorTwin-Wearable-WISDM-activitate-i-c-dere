# Data Card — S6 SensorTwin (WISDM)

## Identificare
- **Dataset:** WISDM Smartphone and Smartwatch Activity and Biometrics Dataset
- **Sursă oficială:** UCI Machine Learning Repository
- **DOI:** https://doi.org/10.24432/C5HK59
- **Versiune:** dataset UCI donat în 2019
- **Utilizare propusă:** clasificarea activităților din semnale inerțiale; etapa de cădere este o extensie separată.

## Conținut și proveniență
- **Subiecți:** 51.
- **Activități:** 18, conform documentației WISDM.
- **Număr de înregistrări:** 15,630,426.
- **Senzori:** accelerometru și giroscop; date de pe smartphone și smartwatch, stocate în directoare separate.
- **Frecvență de eșantionare:** 20 Hz.
- **Câmpuri:** `subject-id`, `activity-code`, `timestamp`, `x`, `y`, `z`.
- **Perioadă calendaristică de colectare:** nu este precizată în metadatele UCI consultate; anul 2019 indică donația datasetului, nu neapărat perioada măsurătorilor.
- **Rezoluție temporală:** aproximativ 0,05 s între eșantioane la 20 Hz; există și variante segmentate în ferestre de 10 secunde.
- **Unități:** valorile x/y/z sunt citiri inerțiale din fișiere; unitatea fizică trebuie confirmată în descrierea fișierului/senzorului înainte de a o declara drept g sau m/s².
- **Valori lipsă:** pagina UCI indică „No” la existența valorilor lipsă. Pipeline-ul verifică din nou fișierul efectiv, deoarece formatele pot varia.

## Licență și utilizare responsabilă
- **Licență:** Creative Commons Attribution 4.0 International (CC BY 4.0), conform UCI.
- **Condiție:** atribuie datasetul lui Gary Weiss/WISDM și păstrează citarea/licența când redistribui date sau rezultate.
- **Limitare majoră:** WISDM nu este un set dedicat căderilor. Un model antrenat pe activitățile WISDM nu poate fi prezentat ca detector de căderi fără date de cădere etichetate și evaluare separată.

## Protocol de evaluare propus
- **Task CP1:** baseline pentru clasificarea activităților disponibile în fișierul selectat.
- **Ferestre:** 200 eșantioane (10 s la 20 Hz), neoverlap, separat pentru fiecare subiect și activitate.
- **Split:** 80% subiecți train / 20% subiecți test, cu seed 42; niciun subiect nu apare în ambele seturi.
- **Baseline:** Random Forest pe statistici simple (mean, standard deviation, min, max) ale axelor x/y/z.
- **Metrici:** accuracy și macro-F1; la CP2 se adaugă matricea de confuzie.
- **Stare rezultate:** nu există scoruri raportate în CP1; acestea se completează după rularea codului pe fișierul local.
