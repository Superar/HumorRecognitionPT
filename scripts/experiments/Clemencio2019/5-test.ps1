. .\scripts\experiments\Clemencio2019\0-variables.ps1

# ------------- CONTENT FEATURES -------------
Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $CONTENT_FEATURES_PATH + '\test\data.hdf5'

# SVC
$SVC_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $CONTENT_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $CONTENT_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\MultinomialNB'
$MULTINOMIALNB_PREDICTIONS_PATH = $CONTENT_FEATURES_PREDICTIONS + '\MultinomialNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $MULTINOMIALNB_PREDICTIONS_PATH `
    --model $MULTINOMIALNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $CONTENT_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $CONTENT_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- HUMOR FEATURES -------------
Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $HUMOR_FEATURES_PATH + '\test\data.hdf5'

# SVC
$SVC_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $HUMOR_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $HUMOR_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$GAUSSIANNB_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\GaussianNB'
$GAUSSIANNB_PREDICTIONS_PATH = $HUMOR_FEATURES_PREDICTIONS + '\GaussianNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $GAUSSIANNB_PREDICTIONS_PATH `
    --model $GAUSSIANNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $HUMOR_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $HUMOR_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH

# ------------- ALL FEATURES -------------
Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
$TEST_FILE = $ALL_FEATURES_PATH + '\test\data.hdf5'

# SVC
$SVC_MODEL_PATH = $ALL_FEATURES_MODELS + '\SVC'
$SVC_PREDICTIONS_PATH = $ALL_FEATURES_PREDICTIONS + '\SVC.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $SVC_PREDICTIONS_PATH `
    --model $SVC_MODEL_PATH

# SVCLinear
$SVCLINEAR_MODEL_PATH = $ALL_FEATURES_MODELS + '\SVCLinear'
$SVCLINEAR_PREDICTIONS_PATH = $ALL_FEATURES_PREDICTIONS + '\SVCLinear.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $SVCLINEAR_PREDICTIONS_PATH `
    --model $SVCLINEAR_MODEL_PATH

# MultinomialNB
$MULTINOMIALNB_MODEL_PATH = $ALL_FEATURES_MODELS + '\MultinomialNB'
$MULTINOMIALNB_PREDICTIONS_PATH = $ALL_FEATURES_PREDICTIONS + '\MultinomialNB.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output  $MULTINOMIALNB_PREDICTIONS_PATH `
    --model $MULTINOMIALNB_MODEL_PATH

# RandomForest
$RANDOMFOREST_MODEL_PATH = $ALL_FEATURES_MODELS + '\RandomForest'
$RANDOMFOREST_PREDICTIONS_PATH = $ALL_FEATURES_PREDICTIONS + '\RandomForest.json'
python .\main.py -v clemencio test `
    --input $TEST_FILE `
    --output $RANDOMFOREST_PREDICTIONS_PATH `
    --model $RANDOMFOREST_MODEL_PATH
