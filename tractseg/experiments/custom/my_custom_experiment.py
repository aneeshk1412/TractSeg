#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.tract_seg import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]
    EXP_PATH = "/scratch/09355/aneeshks/hcp_exp/"

    DATASET_FOLDER = "/scratch/09355/aneeshks/SegTracts/TrainingData/camcan/"      # name of folder that contains all the preprocessed subjects (each subject has its own folder with the name of the subjectID)
    FEATURES_FILENAME = "peaks/peaks"  # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
    LABELS_FILENAME = 'tractmasks/masks/mask'
    DATASET = 'camcan'
    CLASSES = 'camcan_all_classes'
    NR_OF_CLASSES = 61

    SPATIAL_TRANSFORM = 'SpatialTransformPeaks'

    TRAIN = True
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"  # min | max
    LR_SCHEDULE_PATIENCE = 20
    OPTIMIZER = 'Adamax'
    LOSS_FUNCTION = 'boundary_dou'
    BATCH_SIZE = 8
    BATCH_NORM = False
    WEIGHT_DECAY = 0
    USE_DROPOUT = False
    DROPOUT_SAMPLING = False

    LOAD_WEIGHTS = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "My_experiment/best_weights_ep64.npz")
    WEIGHTS_PATH = "" # "/scratch/09355/aneeshks/hcp_exp/my_custom_experiment_x2/best_weights_ep39.npz"  # if empty string: autoloading the best_weights in get_best_weights_path()

    CV_FOLD = 0

    TRAIN_SUBJECTS = ['CC720180', 'CC221565', 'CC520134', 'CC320478', 'CC120816', 'CC420464', 'CC310142', 'CC620085', 'CC510473', 'CC122405', 'CC520013', 'CC620490', 'CC610178', 'CC723395', 'CC711128', 'CC721291', 'CC510259', 'CC620406', 'CC310052', 'CC510226', 'CC410015', 'CC510474', 'CC320687', 'CC620454', 'CC620193', 'CC320686', 'CC520377', 'CC610099', 'CC310203', 'CC510043', 'CC223085', 'CC310256', 'CC721707', 'CC620121', 'CC420286', 'CC721618', 'CC520147', 'CC520424', 'CC410084', 'CC520042', 'CC520279', 'CC110174', 'CC721392', 'CC620284', 'CC310224', 'CC620264', 'CC520127', 'CC320776', 'CC710446', 'CC620354', 'CC322186', 'CC720071', 'CC620129', 'CC510438', 'CC120319', 'CC610671', 'CC620164', 'CC710486', 'CC412021', 'CC420241', 'CC620114', 'CC321976', 'CC620073', 'CC610212', 'CC320575', 'CC310160', 'CC620479', 'CC620259', 'CC120409', 'CC310407', 'CC610653', 'CC320621']
    VALIDATE_SUBJECTS = ['CC420236', 'CC212153', 'CC321585', 'CC722651', 'CC310214', 'CC320616', 'CC610658', 'CC723197', 'CC320814', 'CC610210', 'CC321000', 'CC410354', 'CC121194', 'CC610051', 'CC210617', 'CC321154', 'CC510115']

    TEST = True

    PRUNE = False
    REMOVE_FRACTION = 0.2
    METHOD = "global_unstructured"
