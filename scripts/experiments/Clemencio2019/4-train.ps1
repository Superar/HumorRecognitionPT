. .\scripts\Clemencio2019\0-variables.ps1

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $CONTENT_FEATURES_PATH + '\train\data.hdf5'

# SVC
$SVC_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\MultinomialNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $MULTINOMIALNB_MODEL_PATH `
    --method 'MultinomialNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $HUMOR_FEATURES_PATH + '\train\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# GaussianNB
$GAUSSIANNB_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\GaussianNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $GAUSSIANNB_MODEL_PATH `
    --method 'GaussianNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $ALL_FEATURES_PATH + '\train\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ALL_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ALL_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $ALL_FEATURES_MODELS + '\MultinomialNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $MULTINOMIALNB_MODEL_PATH `
    --method 'MultinomialNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ALL_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'
