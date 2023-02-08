. .\scripts\experiments\cross_corpus\0-variables.ps1

Write-Host 'HEADLINES CORPUS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\scripts\utils\tsv_to_json.py $HEADLINES_CORPUS_PATH $JSON_HEADLINES_CORPUS_PATH
python .\main.py -v preprocess -i $JSON_HEADLINES_CORPUS_PATH -o $PREPROCESSED_HEADLINES_CORPUS_PATH

Write-Host 'ONE-LINERS CORPUS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\scripts\utils\tsv_to_json.py $ONELINERS_CORPUS_PATH $JSON_ONELINERS_CORPUS_PATH
python .\main.py -v preprocess -i $JSON_ONELINERS_CORPUS_PATH -o $PREPROCESSED_ONELINERS_CORPUS_PATH