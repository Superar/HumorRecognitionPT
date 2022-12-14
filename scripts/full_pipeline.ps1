$CORPUS = 'data\Datasets\Balanceados\all.txt'
$FILENAME = (Get-Item $CORPUS).BaseName

# Preprocess data
$JSON_CORPUS = 'data\preprocessed\' + $FILENAME + '.json'
python .\main.py -vv preprocess -i $CORPUS -o $JSON_CORPUS

# Extract features
$FEATURES = 'data\features\' + $FILENAME
python .\main.py -vv feat -i $JSON_CORPUS -o $FEATURES

# Split train and test
$FEATURES_FILE = $FEATURES + '\data.json'
$SPLITPATH = '.\data\split\all\'
python .\scripts\utils\split_corpus.py $FEATURES_FILE $SPLITPATH

# Train model
$TRAINPATH = $SPLITPATH + 'train.json'
$DATE = (Get-Date -format 'yyyyMMdd_HHmmss')
$MODELPATH = '.\results\models\' + $DATE + '-' + $FILENAME
python .\main.py -vv train -i $TRAINPATH -o $MODELPATH

# Test model
$TESTPATH = $SPLITPATH + 'test.json'
python .\main.py -vv test -i $TESTPATH -m $MODELPATH