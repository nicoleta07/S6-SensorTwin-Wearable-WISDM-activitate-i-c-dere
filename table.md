# State of the Art (SOA) — Human Activity Recognition cu senzori mobili și wearable

Tabelul include lucrări fundamentale și review-uri relevante pentru recunoașterea activităților pe baza accelerometrului și a senzorilor purtabili. Nu toate folosesc exact aceeași versiune WISDM sau același protocol experimental ca proiectul S6. Din acest motiv, scorurile numerice nu sunt puse într-o comparație directă: seturile de date, ferestrele, participanții și split-urile diferă.

| Nr. | Lucrare | Date / domeniu | Reprezentare / metodă discutată | Relevanță pentru S6 |
|---:|---|---|---|---|
| 1 | Kwapisz, Weiss & Moore (2011), *Activity recognition using cell phone accelerometers*, ACM SIGKDD Explorations Newsletter, 12(2), 74–82. [DOI](https://doi.org/10.1145/1964897.1964918) | Accelerometru de telefon; activități zilnice; WISDM timpuriu | Ferestre de semnal și clasificare pe caracteristici de accelerometru | Lucrare de bază pentru HAR cu telefon și pentru motivația utilizării WISDM |
| 2 | Lockhart et al. (2011), *Design Considerations for the WISDM Smart Phone-Based Sensor Mining Architecture*, KDD Workshop on Knowledge Discovery from Sensor Data. [WISDM Lab](https://www.cis.fordham.edu/wisdm/) | Arhitectură de colectare și analiză a datelor de senzori mobili | Considerații de achiziție, procesare și analiză a datelor de smartphone | Context pentru proveniența și organizarea ecosistemului WISDM |
| 3 | Bao & Intille (2004), *Activity Recognition from User-Annotated Acceleration Data*, Pervasive Computing, LNCS 3001. [Springer](https://doi.org/10.1007/978-3-540-24646-6_1) | Accelerometre purtabile și activități cotidiene | Caracteristici din semnale de accelerație și clasificare | Fundament pentru extragerea de caracteristici din ferestre |
| 4 | Ravi et al. (2005), *Activity Recognition from Accelerometer Data*, Proceedings of the 17th Conference on Innovative Applications of Artificial Intelligence (IAAI) | Date de accelerometru | Compară abordări de clasificare pentru activități | Context pentru baseline-uri clasice de machine learning |
| 5 | Lara & Labrador (2013), *A Survey on Human Activity Recognition using Wearable Sensors*, IEEE Communications Surveys & Tutorials, 15(3). [DOI](https://doi.org/10.1109/SURV.2012.110112.00192) | Senzori wearable; HAR | Review al senzorilor, caracteristicilor și clasificatorilor | Ajută la motivarea alegerii accelerometrului și a clasificatorilor |
| 6 | Bulling, Blanke & Schiele (2014), *A Tutorial on Human Activity Recognition Using Body-Worn Inertial Sensors*, ACM Computing Surveys, 46(3). [DOI](https://doi.org/10.1145/2499621) | Senzori inerțiali purtabili | Tutorial privind preprocesarea, segmentarea și recunoașterea activităților | Susține deciziile privind ferestrele și procesarea semnalelor |
| 7 | Ramanujam, Perumal & Padmavathi (2021), *Human Activity Recognition With Smartphone and Wearable Sensors Using Deep Learning Techniques: A Review*, IEEE Sensors Journal, 21(12), 13029–13040. [DOI](https://doi.org/10.1109/JSEN.2021.3069927) | Smartphone și wearable | Review al metodelor deep learning pentru HAR | Context pentru utilizarea rețelei neuronale MLP și alternative DL |
| 8 | Straczkiewicz, James & Onnela (2021), *A systematic review of smartphone-based human activity recognition methods for health research*, npj Digital Medicine, 4, 148. [DOI](https://doi.org/10.1038/s41746-021-00514-4) | Smartphone HAR pentru cercetare în sănătate | Review sistematic al senzorilor, preprocesării și clasificatorilor | Evidențiază importanța diversității participanților și a validării |
| 9 | Zhang et al. (2022), *Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances*, Sensors, 22(4), 1476. [DOI](https://doi.org/10.3390/s22041476) | Senzori wearable | Review al arhitecturilor deep learning pentru HAR | Oferă context pentru dezvoltarea ulterioară cu CNN/RNN |
| 10 | Dentamaro et al. (2024), *Human activity recognition with smartphone-integrated sensors: A survey*, Expert Systems with Applications, 246, 123143. [DOI](https://doi.org/10.1016/j.eswa.2024.123143) | Senzori integrați în smartphone | Survey al dataseturilor, senzorilor, setărilor și scorurilor HAR | Sursă recentă pentru compararea reprezentărilor și a protocolului |

## Sinteza literaturii și poziționarea proiectului

Literatura HAR arată două familii importante de abordări:

1. **Machine learning cu feature engineering:** semnalul este segmentat în ferestre, apoi transformat în caracteristici statistice sau de frecvență; clasificatori precum Random Forest, SVM sau alte metode clasice folosesc vectorul de caracteristici.
2. **Deep learning:** rețelele pot învăța reprezentări direct din secvențe sau din caracteristici prelucrate. În această familie intră CNN, RNN și arhitecturi hibride.



Alternativa propusă pentru un experiment viitor este folosirea ferestrei brute de 200 × 3 eșantioane și evaluarea unui CNN 1D. Aceasta nu trebuie prezentată ca rezultat deja obținut.

**Notă metodologică:** tabelul este o sinteză bibliografică, nu un clasament de performanță. Nu se compară numeric Accuracy/F1 între articole fără a verifica faptul că au utilizat același dataset, aceleași clase, aceeași segmentare și același protocol de testare.
