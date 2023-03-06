# Features (Gonçalo Oliveira et al., 2020)

This is a detailed description of the arguments and their expected format used by the model `clemencio` in our implementation.

## `--tfidf`, `--max_tfidf`, `--vectorizer`, and `--ngram`

`--tf-idf` is a flag. If present, TF-IDF counts for the tokens will be calculated.

`--max_tfidf` is an integer, which defines the maximum amount of features to be included with tf-idf. The default value is 1000.

`--vectorizer` is optional in this case. It is a pickled verson of sklearn's `TfIdfVectorizer`. If provided, it will be used to calculate the counts, and no new vectorizer will be fitted.

`--ngram` has a fixed set of possibilities. It represents which combinations of n-grams will be considered for calculating the TF-IDF counts. For example: '1+2+3' will calculate the counts for 1-grams, 2-grams, and 3-grams; meanwhile, '1' will only consider 1-grams.

## ``--sentlex``

This argument requires the path for a sentiment lexicon in JSON. The format is as follows:

```json
{
    "abanou":{
        "Polarity":"1"
    },
    "abarcante":{
        "Polarity":"0"
    },
    "abarcantes":{
        "Polarity":"0"
    },
    "abarrotada":{
        "Polarity":"-1"
    }
}
```

In [`scripts/utils/sentlex_to_json.py`](https://github.com/Superar/HumorRecognitionPT/blob/master/scripts/utils/sentlex_to_json.py), we provide a way to convert from the SentLex format to the expected JSON. For reference, the SentLex format is as follows:

```txt
abanou,abanar.PoS=V;Flex=J:4s|P:3s;TG=HUM:N0:N1;POL:N0=1;POL:N1=-1;ANOT=MAN
abarcante,abarcante.PoS=Adj;FLEX=fs|ms;TG=HUM:N0;POL:N0=0;ANOT=MAN;REV:POL
abarcantes,abarcante.PoS=Adj;FLEX=fp|mp;TG=HUM:N0;POL:N0=0;ANOT=MAN;REV:POL
abarrotada,abarrotado.PoS=Adj;FLEX=fs;TG=HUM:N0;POL:N0=-1;ANOT=JALC
```

## `--slang`

This argument requires the path for a slang lexicon in JSON. The format is as follows:

```json
{
    "0":"barbie",
    "1":"arcaboiço",
    "2":"bitaites",
    "3":"canzana",
    "4":"carcanhois",
    "5":"cunilingus",
    "6":"langonho",
    "7":"narsa"
}
```

If you have a plain text file with a list of slang words (one in a line), we provide a script to convert it to the corresponding JSON in [`scripts/utils/list_to_json.py`](https://github.com/Superar/HumorRecognitionPT/blob/master/scripts/utils/list_to_json.py).

## `--alliteration`

This argument is a flag. If present, the alliteration features will be calculated.


## `--antonym`

This argument requires an antonymy lexicon in JSON. The format is as follows:

```json
[
    {
        "Triple":"piedade ANTONIMO_N_DE impiedade",
        "Number of sources":2,
        "Argument 1":"piedade",
        "Relation":"ANTONIMO_N_DE",
        "Argument 2":"impiedade"
    },
    {
        "Triple":"vetar ANTONIMO_V_DE permitir",
        "Number of sources":2,
        "Argument 1":"vetar",
        "Relation":"ANTONIMO_V_DE",
        "Argument 2":"permitir"
    },
    {
        "Triple":"semelhante ANTONIMO_ADJ_DE desigual",
        "Number of sources":2,
        "Argument 1":"semelhante",
        "Relation":"ANTONIMO_ADJ_DE",
        "Argument 2":"desigual"
    }
]
```

The most important elements are `Argument 1` and `Argument 2`.

If you have a list of triples (as in OntoPT), in the following format, we provide a script in [`scripts/utils/triples_to_json.py`](https://github.com/Superar/HumorRecognitionPT/blob/master/scripts/utils/triples_to_json.py) to convert it to the required JSON format.

```txt
piedade ANTONIMO_N_DE impiedade	2
vetar ANTONIMO_V_DE permitir	2
semelhante ANTONIMO_ADJ_DE desigual	2
```

## `--embeddings`

This argument requires a file in gensim's word2vec keyed vectors format. Such files for Portuguese can be obtained from [NILC Embeddings](www.nilc.icmc.usp.br/embeddings).

## `--mwp`

This argument requires a imageability and concreteness (such as Minho Word Pool) in JSON format, as follows:

```json
{
    "Word (Portuguese)":
    {
        "0":"aba",
        "1":"abacate",
        "2":"abacaxi"
    },
    "Imag_M":
    {
        "0":4.0877192982,
        "1":5.4032258065,
        "2":5.65625
    },
    "Conc_M":
    {
        "0":4.5454545455,
        "1":6.6603773585,
        "2":6.75
    }
}
```

If you have the original Excel in the MWP format, we provide the [`scripts/utils/mwp_to_json.py`](https://github.com/Superar/HumorRecognitionPT/blob/master/scripts/utils/mwp_to_json.py) script to convert it to the requires JSON format.

## `--ner`

This argument is a flag. If it is present, the NER-related features will be calculated. We highlight that this requires the input data to have NER annotations (preprocessing step of our pipeline).

## `--ambiguity`

This argument is a flag. If it is present, the ambiguity features will be calculated using the NLTK interface to OpenWordNet-PT.
