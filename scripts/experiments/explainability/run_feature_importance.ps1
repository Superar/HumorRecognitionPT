# Content features
python .\scripts\experiments\explainability\feature_importance.py `
    .\data\Humor\features\all_content_features\test\data.hdf5 `
    .\results\models\all_content_features\SVCLinear\ `
    .\results\explainability\all_content_features_SVCLinear.ods

# Humor features
python .\scripts\experiments\explainability\feature_importance.py `
    .\data\Humor\features\all_humor_features\test\data.hdf5 `
    .\results\models\all_humor_features\RandomForest\ `
    .\results\explainability\all_humor_features_RandomForest.ods

# All features
python .\scripts\experiments\explainability\feature_importance.py `
    .\data\Humor\features\all_all_features\test\data.hdf5 `
    .\results\models\all_all_features\RandomForest\ `
    .\results\explainability\all_all_features_RandomForest.ods