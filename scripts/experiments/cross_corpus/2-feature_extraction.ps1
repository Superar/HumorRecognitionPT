. .\scripts\experiments\cross_corpus\0-variables.ps1

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n

Write-Host 'TRAIN CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\main.py -v feat -i $PREPROCESSED_HEADLINES_CORPUS_PATH -o $HEADLINES_CONTENT_FEATURES_PATH --tfidf
python .\main.py -v feat -i $PREPROCESSED_ONELINERS_CORPUS_PATH -o $ONELINERS_CONTENT_FEATURES_PATH --tfidf

Write-Host 'CROSS CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
$HEADLINES_VECTORIZER_PATH = $HEADLINES_CONTENT_FEATURES_PATH + '\vectorizer.pkl'
$ONELINERS_VECTORIZER_PATH = $ONELINERS_CONTENT_FEATURES_PATH + '\vectorizer.pkl'

python .\main.py -v feat -i $PREPROCESSED_HEADLINES_CORPUS_PATH `
    -o $HEADLINES_CONTENT_FEATURES_ONELINERS_PATH `
    --tfidf --vectorizer $ONELINERS_VECTORIZER_PATH

python .\main.py -v feat -i $PREPROCESSED_ONELINERS_CORPUS_PATH `
    -o $ONELINERS_CONTENT_FEATURES_HEADLINES_PATH `
    --tfidf --vectorizer $HEADLINES_VECTORIZER_PATH

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n

Write-Host 'HEADLINES CORPUS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\main.py -v feat -i $PREPROCESSED_HEADLINES_CORPUS_PATH -o $HEADLINES_HUMOR_FEATURES_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

Write-Host 'ONE-LINERS CORPUS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\main.py -v feat -i $PREPROCESSED_ONELINERS_CORPUS_PATH -o $ONELINERS_HUMOR_FEATURES_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n

Write-Host 'TRAIN CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\main.py -v feat -i $PREPROCESSED_HEADLINES_CORPUS_PATH -o $HEADLINES_ALL_FEATURES_PATH `
    --tfidf `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

python .\main.py -v feat -i $PREPROCESSED_ONELINERS_CORPUS_PATH -o $ONELINERS_ALL_FEATURES_PATH `
    --tfidf `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

Write-Host 'CROSS CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

$HEADLINES_VECTORIZER_PATH = $HEADLINES_ALL_FEATURES_PATH + '\vectorizer.pkl'
$ONELINERS_VECTORIZER_PATH = $ONELINERS_ALL_FEATURES_PATH + '\vectorizer.pkl'

python .\main.py -v feat -i $PREPROCESSED_HEADLINES_CORPUS_PATH -o $HEADLINES_ALL_FEATURES_ONELINERS_PATH `
    --tfidf `
    --vectorizer $ONELINERS_VECTORIZER_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity

python .\main.py -v feat -i $PREPROCESSED_ONELINERS_CORPUS_PATH -o $ONELINERS_ALL_FEATURES_HEADLINES_PATH `
    --tfidf `
    --vectorizer $HEADLINES_VECTORIZER_PATH `
    --sentlex $SENTLEX `
    --slang $SLANG `
    --alliteration `
    --antonym $ANTONYM_TRIPLES `
    --embeddings $EMBEDDINGS `
    --mwp $MWP `
    --ner `
    --ambiguity