# Content features
python .\scripts\experiments\explainability\feature_importance\shap_kernel_feature_importance.py `
    .\data\Humor\features\all_content_features\train\data.hdf5 `
    .\data\Humor\features\all_content_features\test\data.hdf5 `
    .\results\models\all_content_features\SVCLinear `
    .\results\explainability\all_content_features_SVCLinear

# Humor features

python .\scripts\experiments\explainability\feature_importance\shap_callable_feature_importance.py `
    .\data\Humor\features\all_humor_features\test\data.hdf5 `
    .\results\models\all_humor_features\RandomForest `
    .\results\explainability\all_humor_features_RandomForest

# All features

python .\scripts\experiments\explainability\feature_importance\shap_callable_feature_importance.py `
    .\data\Humor\features\all_all_features\test\data.hdf5 `
    .\results\models\all_all_features\RandomForest `
    .\results\explainability\all_all_features_RandomForest
