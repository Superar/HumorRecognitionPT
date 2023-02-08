. .\scripts\experiments\cross_corpus\0-variables.ps1

Write-Host 'HEADLINES -> ONE-LINERS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $ONELINERS_CONTENT_FEATURES_HEADLINES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $HEADLINES_ONELINERS_CONTENT_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $HEADLINES_ONELINERS_CONTENT_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\MultinomialNB'
$MULTINOMIALNB_PREDICTIONS_PATH = $HEADLINES_ONELINERS_CONTENT_FEATURES_PREDICTIONS + '\MultinomialNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $MULTINOMIALNB_PREDICTIONS_PATH `
    --model $MULTINOMIALNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HEADLINES_CONTENT_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $HEADLINES_ONELINERS_CONTENT_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $ONELINERS_HUMOR_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $HEADLINES_ONELINERS_HUMOR_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $HEADLINES_ONELINERS_HUMOR_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$GAUSSIANNB_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\GaussianNB'
$GAUSSIANNB_PREDICTIONS_PATH = $HEADLINES_ONELINERS_HUMOR_FEATURES_PREDICTIONS + '\GaussianNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $GAUSSIANNB_PREDICTIONS_PATH `
    --model $GAUSSIANNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HEADLINES_HUMOR_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $HEADLINES_ONELINERS_HUMOR_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $ONELINERS_ALL_FEATURES_HEADLINES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $HEADLINES_ONELINERS_ALL_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $HEADLINES_ONELINERS_ALL_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\MultinomialNB'
$MULTINOMIALNB_PREDICTIONS_PATH = $HEADLINES_ONELINERS_ALL_FEATURES_PREDICTIONS + '\MultinomialNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $MULTINOMIALNB_PREDICTIONS_PATH `
    --model $MULTINOMIALNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HEADLINES_ALL_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $HEADLINES_ONELINERS_ALL_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- TRANSFORMER -------------
Write-Host 'TRANSFORMER' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRANSFOMER_MODEL_PATH = $HEADLINES_TRANSFORMER_MODEL + '\checkpoint-1500'
python .\main.py transformer test `
    --input $PREPROCESSED_ONELINERS_CORPUS_PATH `
    --output $HEADLINES_ONELINERS_TRANSFORMER_PREDICTIONS `
    --model $TRANSFOMER_MODEL_PATH

Write-Host 'ONE-LINERS -> HEADLINES' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $HEADLINES_CONTENT_FEATURES_ONELINERS_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $ONELINERS_HEADLINES_CONTENT_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $ONELINERS_HEADLINES_CONTENT_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\MultinomialNB'
$MULTINOMIALNB_PREDICTIONS_PATH = $ONELINERS_HEADLINES_CONTENT_FEATURES_PREDICTIONS + '\MultinomialNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $MULTINOMIALNB_PREDICTIONS_PATH `
    --model $MULTINOMIALNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ONELINERS_CONTENT_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $ONELINERS_HEADLINES_CONTENT_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $HEADLINES_HUMOR_FEATURES_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $ONELINERS_HEADLINES_HUMOR_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $ONELINERS_HEADLINES_HUMOR_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$GAUSSIANNB_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\GaussianNB'
$GAUSSIANNB_PREDICTIONS_PATH = $ONELINERS_HEADLINES_HUMOR_FEATURES_PREDICTIONS + '\GaussianNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $GAUSSIANNB_PREDICTIONS_PATH `
    --model $GAUSSIANNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ONELINERS_HUMOR_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $ONELINERS_HEADLINES_HUMOR_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $HEADLINES_ALL_FEATURES_ONELINERS_PATH + '\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $ONELINERS_HEADLINES_ALL_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $ONELINERS_HEADLINES_ALL_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\MultinomialNB'
$MULTINOMIALNB_PREDICTIONS_PATH = $ONELINERS_HEADLINES_ALL_FEATURES_PREDICTIONS + '\MultinomialNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $MULTINOMIALNB_PREDICTIONS_PATH `
    --model $MULTINOMIALNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ONELINERS_ALL_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $ONELINERS_HEADLINES_ALL_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- TRANSFORMER -------------
Write-Host 'TRANSFORMER' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TRANSFOMER_MODEL_PATH = $ONELINERS_TRANSFORMER_MODEL + '\checkpoint-500'
python .\main.py transformer test `
    --input $PREPROCESSED_HEADLINES_CORPUS_PATH `
    --output $ONELINERS_HEADLINES_TRANSFORMER_PREDICTIONS `
    --model $TRANSFOMER_MODEL_PATH
