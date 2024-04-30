
class paths:
    MODEL_WEIGHTS = "/content/efficientnet/outputs/tf_efficientnet_b0_epoch_3_avg_train_loss_0.32836671578192084.pth"
    OUTPUT_DIR = "/content/working/"
    TEST_CSV = "/content/gcs/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS= "/content/gcs/hms-harmful-brain-activity-classification/test_eegs/"
    TRAIN_CSV = "/content/gcs/hms-harmful-brain-activity-classification/cleaned_train.csv"
    TRAIN_EEGS= "/content/gcs/hms-harmful-brain-activity-classification/train_eegs/"
    TRAIN_SPECTROGRAMS = "/content/gcs/hms-harmful-brain-activity-classification/train_spectrograms/"
    FEATURE_FOLDER = "/content/gcs/hms-harmful-brain-activity-classification/"
    COMBINED_FEATURES = "/content/gcs/hms-harmful-brain-activity-classification/combined_sorted_features.csv"
    FUSION_DATA = "/content/gcs/hms-harmful-brain-activity-classification/cleaned_train.csv"
    FINAL_FEATURE_FOLDER = "/content/gcs/features/"
    ROCKET_DIR = "/content/gcs/models/rocket/"
    XG_MODEL = "/content/gcs/models/xgboost_model.pkl"
    SAVE_PATH = "/content/gcs/models/"

    @classmethod
    def prepend_path_prefix(cls, prefix):
        for attr_name in dir(cls):
            if not attr_name.startswith("__") and isinstance(getattr(cls, attr_name), str):
                setattr(cls, attr_name, prefix + getattr(cls, attr_name))

class config:
    BATCH_SIZE = 4
    MODEL = "tf_efficientnet_b0"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    N_EPOCHS = 15
    LR = 1e-3
    LABEL_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    EXCESS = 0.999