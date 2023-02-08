. .\scripts\experiments\cross_corpus\0-variables.ps1

Write-Host 'HEADLINES CORPUS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $HEADLINES_CONTENT_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\MultinomialNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $MULTINOMIALNB_MODEL_PATH `
    --method 'MultinomialNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $HEADLINES_HUMOR_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# GaussianNB
$GAUSSIANNB_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\GaussianNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $GAUSSIANNB_MODEL_PATH `
    --method 'GaussianNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $HEADLINES_ALL_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\MultinomialNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $MULTINOMIALNB_MODEL_PATH `
    --method 'MultinomialNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- TRANSFORMER -------------
Write-Host 'TRANSFOMER' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\main.py transformer fine-tune `
    --input $PREPROCESSED_HEADLINES_CORPUS_PATH `
    --output $HEADLINES_TRANSFORMER_MODEL

Write-Host 'ONE-LINERS CORPUS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $ONELINERS_CONTENT_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\MultinomialNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $MULTINOMIALNB_MODEL_PATH `
    --method 'MultinomialNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $ONELINERS_HUMOR_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# GaussianNB
$GAUSSIANNB_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\GaussianNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $GAUSSIANNB_MODEL_PATH `
    --method 'GaussianNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRAIN_FILE = $ONELINERS_ALL_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\SVC'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVC_MODEL_PATH `
    --method 'SVC'

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\SVCLinear'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $SVCLINEAR_MODEL_PATH `
    --method 'SVCLinear'

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\MultinomialNB'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $MULTINOMIALNB_MODEL_PATH `
    --method 'MultinomialNB'

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\RandomForest'
python .\main.py -v clemencio train `
    --input $TRAIN_FILE `
    --output  $RANDOMFOREST_MODEL_PATH `
    --method 'RandomForest'

# ------------- TRANSFORMER -------------
Write-Host 'TRANSFOMER' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\main.py transformer fine-tune `
    --input $PREPROCESSED_ONELINERS_CORPUS_PATH `
    --output $ONELINERS_TRANSFORMER_MODEL