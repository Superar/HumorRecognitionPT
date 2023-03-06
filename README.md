# Humor Recognition in Portuguese

Humor Recognition and Machine Learning Explainability implementations for the paper **``What do Humor Classifiers Learn? An Attempt to Explain Humor Recognition Models**.''

## Installation

This project is available as a python module to make installation and re-use easier.

Using Pipenv, just run:
```shell
$ pipenv install
```

If you prefer to use pip, run:
```shell
$ pip install -r requirements.txt
$ pip install .
```

It is also necessary to install some nltk packages, in the `python` terminal, run:

```python
import nltk
nltk.download('punkt')
nltk.download('floresta')
```

## Folder structure

Here is a general view of the structure of this project, including a brief description of the contents in each folder.

```bash
HumorRecognitionPT
│
├───data # Data folder. Not included in this repo
├───docs # Extra documentation
├───results
│   ├───explainability # Explainability graphs
│   ├───models # Trained models. Not included in this repo
│   └───predictions # Test prediction for each trained model
├───scripts
│   ├───experiments # Commands used for every experiment (in Powershell)
│   │   ├───Clemencio2019
│   │   ├───Clemencio2019_variation
│   │   ├───cross_corpus
│   │   ├───explainability
│   │   │   ├───clustering
│   │   │   └───feature_importance
│   │   └───transformers
│   └───utils # Extra scripts
├───src # Source code
│   ├───commands # Command and argument definitions
│   ├───features # Feature-Extraction implementations
│   ├───methods # Humor Recognition models implementations
│   ├───utils # General-usage functions
└───tests
```

## Running

The script to be run is `main.py`. It has different commands that must be specified one at a time:

```shell
$ python main.py [-h] [--verbose] {preprocess,feature-extraction,feat,clemencio,transformer}
```

- `preprocess` -- Preprocessing process required for the method `clemencio`. Tokenization, POS Tagging, NER, and Lemmatization.

```shell
$ python main.py preprocess [-h] --input INPUT [--output OUTPUT]
```

- `feature-extraction` or `feat` -- Feature extraction required for the method `clemencio`. It requires various arguments for calculating different kinds of features. For more information about these arguments we refer to [`docs\FEATURES.md`](https://github.com/Superar/HumorRecognitionPT/blob/master/docs/FEATURES.md).

```shell
$ python main.py feature-extraction [-h] --input INPUT [--output OUTPUT] [--tfidf]
                                    [--max_tfidf MAX_TFIDF] [--vectorizer VECTORIZER]
                                    [--ngram {1,2,3,1+2,2+3,1+2+3}] [--sentlex SENTLEX]
                                    [--slang SLANG] [--alliteration] [--antonym ANTONYM]
                                    [--embeddings EMBEDDINGS] [--mwp MWP] [--ner]
                                    [--ambiguity]
```
- `clemencio` -- A re-implementation of the original method described by Gonçalo Oliveira et al. (2020) and Clemêncio (2019). It has two possible sub-commands: for training and for testing the model.

```shell
$ python main.py clemencio [-h] {train,test}
$ python main.py clemencio train [-h] --input INPUT [--output OUTPUT]
                                 [--method {SVC,SVCLinear,MultinomialNB,GaussianNB,RandomForest}]
$ python main.py clemencio test [-h] --input INPUT --model MODEL [--output OUTPUT]
```

- `transformer` -- An implementation to fine-tune a HuggingFace model for Humor Recognition as a classification task.

```shell
$ python main.py transformer [-h] {fine-tune,test}
$ python main.py transformer fine-tune [-h] --input INPUT --output OUTPUT [--model MODEL]
$ python main.py transformer test [-h] --input INPUT [--model MODEL] [--output OUTPUT]
```
To see examples of working commands, the ones we used for our experiments in the paper, we suggest exploring the Powershell scripts in [`experiments`](https://github.com/Superar/HumorRecognitionPT/tree/master/scripts/experiments).

