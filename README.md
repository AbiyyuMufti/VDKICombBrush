# VDKICombBrush

## Link:
https://github.com/AbiyyuMufti/VDKICombBrush

## Teilnehmer:
- ### Muhammad Abiyyu Mufti Hanif - 60750 - hamu1013@h-ka.de
- ### Davin Farrell Lukito -75420 - luda1013@h-ka.de

## Wichtige Ordner-Inhalt bzw. Dateien:
- #### resources: Bilder für Bürste und Kämme

- #### preprocessing: Package um die rohe Bilder zu bearbeiten (z.B. Änderung der Bildgröße), die Merkmale zu extrahieren und zu einer Tabel oder csv-Datei zu speichern
      - feature_extraction.py = Funktionen zum Extrahieren von Merkmalen mittels opencv und numpy z.B. *Corner Detection with Haris* oder *Contour Detection*
      - image_preprocessing.py = Aufrufen Funktionen aus feature_extraction.py, Benennen alle Merkmalem und Speichern in Form einer Tabelle oder csv-Datei
      - img_to_array.py = Anpassen von Eigenschaften der Bilder bzgl. für CNN-Verfahren
      - simpledatasetloader.py = Aufladen alle Bilder und es lässt sich alle Merkmallen für alle Bilder in _Path_ extrahieren
      
- #### our_own_ai_process:
      - ai_process.py: *Base classes* für unsere eigene AI-Verfahren
      - own_cnn.py: *Convolutional Neural Network* -Verfahren mittels vorgefertigtem Framework: Tensorflow und Keras
      - own_decision_tree.py: Funktionen für Entscheidungsbaum-Algorithmus
      - own_knn.py: Funktionen für *K-Nearest-Neighbor* -Algorithmus
      - own_random_forest.py: Funktionen für *Random-Forest* -Algorithmus
      - utility.py: gemeinsame von einigen Verfahren verwendete Funktionen, z.B. Funktion fürs Teilen der Data zum Trainieren und Testen
      
- #### DataSetCreator.ipynb: [Jupyter-Notebook-Datei]: Aufladen von allen Bilder, Extrahieren der Merkmale und Exportieren zu CSV oder *panda-dataframe*

- #### ImplementationAI.ipynb: [Jupyter-Notebook-Datei]: Ausführung unserer 4 Verfahren bzw. Anzeigen der *Accuracy*, *Precision* und auch *Recall* und *Support*. Bei CNN wird zusätzlich die "Training Loss" und "Accuracy" in Form einer Grafik angezeigt  
       
- #### feature_extraction.ipynb: [Jupyter-Notebook-Datei]: Aufladen von einem Bild und seine Merkmale lassen sich durch die package extrahieren

## Ausführung der Verfahren:
- Öffnen der ipynb-Datei: ImplementationAI.ipynb in Jupyter-Notebook
- Lassen Sie das Script durchführen
- Anmerkung für Decision Tree: der Baum bzw. der Afbau ist nicht gut angezeigt in ipynb; auch trotz mit dem Befehl pprint.pprint(ODT.tree) oder ODT.plot_tree(). Der Baum ist viel besser angezeigt, in dem man die Datei: own_decision_tree.py ausführt.
    
## Einige Dateien zu checken oder anzuschauen:
- DataSetCreator.pdf: Durchführung aller Bilder in /resources/
- Features and Scores.pdf: Zusammenfassung der extrahierten Merkmale und Ergebnis-vergleich zwischen Decision Tree, Random Forest, KNN und CNN
- ImplementationAI.pdf: Durchführung aller 4 AI-Verfahren
- Scores.xlsx: Liste der extrahierten Merkmale, Ergebnisbeispiel beim Extrahieren, Vergleich der Ergebnisse zwischen Decision Tree, Random Forest, KNN und CNN

  
