. .\scripts\experiments\Clemencio2019\0-variables.ps1

python .\scripts\utils\tsv_to_json.py $ORIGINAL_CORPUS_PATH $JSON_CORPUS_PATH
python .\main.py -v preprocess -i $JSON_CORPUS_PATH -o $PREPROCESSED_CORPUS_PATH
