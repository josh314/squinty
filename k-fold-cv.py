import numpy as np

predictor_folds = np.array_split(train_predictor_data,10)
target_folds = np.array_split(train_target_data,10)
