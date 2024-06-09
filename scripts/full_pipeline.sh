CORPUS="data/Humor/Datasets/Balanceados/all.json"
FILENAME="$(basename -- $CORPUS .json)"

# Preprocess data
echo "------------ PREPROCESSING ------------"
JSON_CORPUS="data/Humor/preprocessed/${FILENAME}.json"
python ./main.py -vv preprocess -i $CORPUS -o $JSON_CORPUS

# Extract features
echo "------------ FEATURES ------------"
FEATURES="data/features/$FILENAME"
python ./main.py -vv feat -i $JSON_CORPUS -o $FEATURES

# Split train and test
echo "------------ SPLIT ------------"
FEATURES_FILE="${FEATURES}/data.hdf5"
SPLITPATH="./data/split/all"
python ./scripts/utils/split_corpus.py $FEATURES_FILE $SPLITPATH

# Train model
echo "------------ TRAIN ------------"
# TRAINPATH="${SPLITPATH}/train.hdf5"
printf -v DATE "%(%Y%m%d_%H%M%S)T" -1
MODELPATH="./results/models/${DATE}-${FILENAME}"
python ./main.py -vv train -i $TRAINPATH -o $MODELPATH

# Test model
Write-Output '------------ TEST ------------'
$TESTPATH = $SPLITPATH + 'test.hdf5'
python .\main.py -vv test -i $TESTPATH -m $MODELPATH
