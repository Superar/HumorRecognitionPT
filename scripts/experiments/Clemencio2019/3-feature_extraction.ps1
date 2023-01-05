. .\scripts\Clemencio2019\0-variables.ps1

$TRAIN_FILE = $SPLIT_CORPUS_PATH + '\train.json'
$TEST_FILE = $SPLIT_CORPUS_PATH + '\test.json'

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FEATURES_PATH = $CONTENT_FEATURES_PATH + '\train'
$TEST_FEATURES_PATH = $CONTENT_FEATURES_PATH + '\test'
$VECTORIZER_PATH = $TRAIN_FEATURES_PATH + '\vectorizer.pkl'
python .\main.py -v feat -i $TRAIN_FILE -o $TRAIN_FEATURES_PATH --tfidf
python .\main.py -v feat -i $TEST_FILE -o $TEST_FEATURES_PATH --tfidf --vectorizer $VECTORIZER_PATH

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FEATURES_PATH = $HUMOR_FEATURES_PATH + '\train'
$TEST_FEATURES_PATH = $HUMOR_FEATURES_PATH + '\test'
python .\main.py -v feat -i $TRAIN_FILE -o $TRAIN_FEATURES_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

python .\main.py -v feat -i $TEST_FILE -o $TEST_FEATURES_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

# ------------- HUMOR FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FEATURES_PATH = $ALL_FEATURES_PATH + '\train'
$TEST_FEATURES_PATH = $ALL_FEATURES_PATH + '\test'
$VECTORIZER_PATH = $TRAIN_FEATURES_PATH + '\vectorizer.pkl'

python .\main.py -v feat -i $TRAIN_FILE -o $TRAIN_FEATURES_PATH `
    --tfidf `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

python .\main.py -v feat -i $TEST_FILE -o $TEST_FEATURES_PATH `
    --tfidf `
    --vectorizer $VECTORIZER_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity