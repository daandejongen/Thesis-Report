*Daan de Jong
Master Thesis Methodology and Statistics for the Biomedical, Behavioral and Social Sciences
Negation Scope Detection with a BiLSTM NN*

------------------------------------------------------------------------------------------

The Thesis-Report repository contains two folders: *Text* and *Data and code*.

## Text
- img folder:
	* `Plaatjesmaker.ppt` and `Plaatjesmaker2.ppt` were used to create model.png, model2.png and prediction types.png
	* `model.png` is outdated
	* `model2.png` was used in the final Report (by `main.tex` in the text folder)
	* `prediction types.png` was used in the final Report (by `main.tex` in the text folder)
- `main.tex` contains the text that generates the final report
- `refs_report` is a `bib` file that contains the references used by main.tex
- `title page.tex` is used by `main.tex`, to split up the first page from the actual content of the report
- `Draft 1 Report DDJ.pdf` is the main draft which was feedbacked by Ayoub Bagheri and Joost de Jong (feedback not included)
- `Research Report Daan de Jong.pdf` is the final Report that was handed in
- `main.bbl`, `main.blg`, `main.log`, `main.synctex.gz`, `textut.log`, `title page.log` are byproducts of `main.tex` and `title page.tex`

## Data and Code
- `__pycache__` is a folder that contains byproducts of the `.py` files
- Original is a folder that contains all data used in this study
	- freely downloadable [here](https://www.kaggle.com/ma7555/bioscope-corpus-negation-annotated)
	- the folder bioscope was not used
	- `bioscope_abstract.csv` and `bioscope_full.csv` are the data files that were used in this study (by `Preprocessing.py`)
- All `.csv` files here are rendered by `Preprocessing.py`, and are read in `Model.py`, except for `E.csv` (the word embeddings, from `word2vec.py`)
- `CustomEmbedding.py` defines a [keras](https://keras.io) layer class, to be used in `Model.py`
- `CustomLoss.py` defines the loss function used in the study, used by `Model.py`
- `CustomMetric.py` defines the metrics used in the study, used by `Model.py`
- `Data description table.py` renders the information used in Table 1 of the Report
- `Model.py` defines, compiles, trains, tests and evaluates the model
- `Preprocessing.py` bridges the gap between the raw data `.csv` files and the train and test data suited for the model
- `word2vec.py` is an implementation of the [word2vec](https://arxiv.org/pdf/1301.3781.pdf) algorithm and renders `E.csv`, the word embeddings