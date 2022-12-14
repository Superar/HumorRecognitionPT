$CORPUS = 'data\Datasets\Balanceados\all.txt'
$FILENAME = (Get-Item $CORPUS).BaseName

# Preprocess data
Write-Output '------------ PREPROCESSING ------------'
$JSON_CORPUS = 'data\preprocessed\' + $FILENAME + '.json'
python .\main.py -vv preprocess -i $CORPUS -o $JSON_CORPUS

# Extract features
Write-Output '------------ FEATURES ------------'
$FEATURES = 'data\features\' + $FILENAME
python .\main.py -vv feat -i $JSON_CORPUS -o $FEATURES

# Split train and test
Write-Output '------------ SPLIT ------------'
$FEATURES_FILE = $FEATURES + '\data.hdf5'
$SPLITPATH = '.\data\split\all\'
python .\scripts\utils\split_corpus.py $FEATURES_FILE $SPLITPATH

# Train model
Write-Output '------------ TRAIN ------------'
$TRAINPATH = $SPLITPATH + 'train.hdf5'
$DATE = (Get-Date -format 'yyyyMMdd_HHmmss')
$MODELPATH = '.\results\models\' + $DATE + '-' + $FILENAME
python .\main.py -vv train -i $TRAINPATH -o $MODELPATH

# Test model
Write-Output '------------ TEST ------------'
$TESTPATH = $SPLITPATH + 'test.hdf5'
python .\main.py -vv test -i $TESTPATH -m $MODELPATH