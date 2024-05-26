cfg.YOLO.STRIDES = [8, 16, 32]
cfg.YOLO.CLASSES = './annotations/_classes.names'
cfg.YOLO.YAML_PATH = ''
cfg.YOLO.NUM_CLASSES = 1
cfg.YOLO.TOPK = 13
cfg.YOLO.MODEL_TYPE = 's' # [s, m, l]
cfg.YOLO.ANGLE_MIN = -180
cfg.YOLO.ANGLE_MAX = 180

cfg.YOLO.BATCH_SIZE = 4
cfg.YOLO.TRAIN.EPOCHS = 500
# cfg.YOLO.TRAIN.EPOCHS_FOR_PRETRAINING = 4
# cfg.YOLO.TRAIN.EPOCHS_FOR_FINE_TUNING = 200
cfg.YOLO.TRAIN.MODEL_SIZE = (640, 640)
cfg.YOLO.TRAIN.ANNOT_PATH = ''
cfg.YOLO.TRAIN.SAVED_MODEL_DIR = './saved_model'
# cfg.YOLO.TRAIN.SAVED_MODEL_DIR = './adconv_saved_model'
cfg.YOLO.TRAIN.DATA_AUG = True
cfg.YOLO.TRAIN.HORIZONTAL_FLIP = False
cfg.YOLO.TRAIN.VERTICAL_FLIP = False
cfg.YOLO.TRAIN.RANDOM_CROP = True
cfg.YOLO.TRAIN.RANDOM_TRANSLATE = True
cfg.YOLO.TRAIN.RANDOM_ROTATE = False
cfg.YOLO.TRAIN.USE_COLORJITTER = True
cfg.YOLO.TRAIN.LR_INIT = 1e-4
cfg.YOLO.TRAIN.OPTIMIZER_TYPE = 'adam'
cfg.YOLO.TRAIN.VISUAL_LEARNING_PROCESS = True
cfg.YOLO.TRAIN.TRANSFER = 'scratch' #'transfer'
# cfg.YOLO.TRAIN.TRANSFER = 'transfer'
# cfg.YOLO.TRAIN.TRANSFER = 'resume'
cfg.YOLO.TRAIN.ADD_IMG_PATH = ''
cfg.YOLO.TRAIN.LOSS_TYPE_FOR_BBOXES = 'siou' # probiou
cfg.YOLO.TRAIN.CONF_THRESHOLD = 0.5
cfg.YOLO.TRAIN.IOU_THRESHOLD = 0.25
cfg.YOLO.TRAIN.MAX_BBOX_PER_IMAGE = 310
cfg.YOLO.TRAIN.BACKBONE_DATE = None
cfg.YOLO.TRAIN.NECK_DATA = None
cfg.YOLO.TRAIN.PATIENCE = 10
cfg.YOLO.TRAIN.N_SAMPLES_PER_EPOCH = 10000
cfg.YOLO.TRAIN.USE_VALID_DATASET = False
# cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING = './icdar_dataset/icdar17/test/images'
# cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING = './total_text_dataset/Images/Test'
cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING = './dot_matrix_dataset/2023_03_09/Pass'


# generate fake image
cfg.GENERATE_FAKE_IMAGE = edict()
# cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR = './synthtext_images'
cfg.GENERATE_FAKE_IMAGE.IMAGE_BACKGROUND_DIR = './imagenet_100k_512px'
cfg.GENERATE_FAKE_IMAGE.FONT_DIR = "./fonts"
cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MIN = 40
cfg.GENERATE_FAKE_IMAGE.FONT_SIZE_MAX = 250
cfg.GENERATE_FAKE_IMAGE.TEXT_COLOR = None
cfg.GENERATE_FAKE_IMAGE.IMAGE_SIZE = (640, 640)
cfg.GENERATE_FAKE_IMAGE.WORD_COUNT = 8
cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MIN = 5
cfg.GENERATE_FAKE_IMAGE.WORD_LENGTH_MAX = 30
cfg.GENERATE_FAKE_IMAGE.ANGLE_MIN = -180
cfg.GENERATE_FAKE_IMAGE.ANGLE_MAX =  180


# icdar for training
cfg.YOLO.TRAIN.ICDAR13_ANNOT_DIR = 'icdar_dataset/icdar13/Challenge2_Training_Task1_GT'
cfg.YOLO.TRAIN.ICDAR13_IMAGE_DIR_FOR_TRAINING = 'icdar_dataset/icdar13/Challenge2_Training_Task12_Images'
cfg.YOLO.TRAIN.ICDAR15_ANNOT_DIR = 'icdar_dataset/icdar15/ch4_training_localization_transcription_gt'
cfg.YOLO.TRAIN.ICDAR15_IMAGE_DIR_FOR_TRAINING = 'icdar_dataset/icdar15/ch4_training_images'
cfg.YOLO.TRAIN.ICDAR17_ANNOT_DIR = 'icdar_dataset/icdar17/train/gt'
cfg.YOLO.TRAIN.ICDAR17_IMAGE_DIR_FOR_TRAINING = 'icdar_dataset/icdar17/train/images'

# icdar for testing
cfg.YOLO.TRAIN.ICDAR13_TEST_ANNOT_DIR = 'icdar_dataset/icdar13/Challenge2_Test_Task1_GT'
cfg.YOLO.TRAIN.ICDAR13_IMAGE_DIR_FOR_TESTING = 'icdar_dataset/icdar13/Challenge2_Test_Task12_Images'
cfg.YOLO.TRAIN.ICDAR15_TEST_ANNOT_DIR = 'icdar_dataset/icdar15/Challenge4_Test_Task1_GT'
cfg.YOLO.TRAIN.ICDAR15_IMAGE_DIR_FOR_TESTING = 'icdar_dataset/icdar15/ch4_test_images'
cfg.YOLO.TRAIN.ICDAR17_TEST_ANNOT_DIR = 'icdar_dataset/icdar17/test/gt'
cfg.YOLO.TRAIN.ICDAR17_IMAGE_DIR_FOR_TESTING = 'icdar_dataset/icdar17/test/images'

# total text for training
cfg.YOLO.TRAIN.TOTAL_TEXT_ANNOT_DIR = 'total_text_dataset/gt/Train'
cfg.YOLO.TRAIN.TOTAL_TEXT_IMAGE_DIR_FOR_TRAINING = 'total_text_dataset/Images/Train'

# total text for testing
cfg.YOLO.TRAIN.TOTAL_TEXT_TEST_ANNOT_DIR = 'total_text_dataset/gt/Test'
cfg.YOLO.TRAIN.TOTAL_TEXT_IMAGE_DIR_FOR_TESTING = 'total_text_dataset/Images/Test'

# text ocr
cfg.YOLO.TRAIN.TEXT_OCR_TRAIN_ANNOT_PATH = 'textocr/train_annot/TextOCR_0.1_train.json'
cfg.YOLO.TRAIN.TEXT_OCR_VALID_ANNOT_PATH = 'textocr/train_annot/TextOCR_0.1_val.json'
cfg.YOLO.TRAIN.TEXT_OCR_IMAGE_DIR = 'textocr/train_images'


# Pretrain model
train_dataset = YOLODataset()

classes_map = train_dataset.classes

trainer = YOLOTrainer(imgsz=cfg.YOLO.TRAIN.MODEL_SIZE, epochs=cfg.YOLO.TRAIN.EPOCHS, topk=cfg.YOLO.TOPK, batch_size=cfg.YOLO.BATCH_SIZE,
                      saved_model_dir=cfg.YOLO.TRAIN.SAVED_MODEL_DIR, model_type=cfg.YOLO.MODEL_TYPE, transfer=cfg.YOLO.TRAIN.TRANSFER,
                      optimizer_type=cfg.YOLO.TRAIN.OPTIMIZER_TYPE, lr_init=cfg.YOLO.TRAIN.LR_INIT, class_file=cfg.YOLO.CLASSES,
                      iou_threshold=cfg.YOLO.TRAIN.IOU_THRESHOLD, conf_threshold=cfg.YOLO.TRAIN.CONF_THRESHOLD,
                      patience=cfg.YOLO.TRAIN.PATIENCE, iou_type=cfg.YOLO.TRAIN.LOSS_TYPE_FOR_BBOXES,
                      angle_min=cfg.YOLO.ANGLE_MIN, angle_max=cfg.YOLO.ANGLE_MAX,
                      use_valid_dataset=cfg.YOLO.TRAIN.USE_VALID_DATASET , image_dir_for_testing=cfg.YOLO.TRAIN.IMAGE_DIR_FOR_TESTING,
                      visual_learning_process=cfg.YOLO.TRAIN.VISUAL_LEARNING_PROCESS)

model = trainer.train(train_dataset)