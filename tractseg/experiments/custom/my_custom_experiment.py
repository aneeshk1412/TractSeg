#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tractseg.experiments.tract_seg import Config as TractSegConfig


class Config(TractSegConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATASET_FOLDER = "/Users/aneeshks/Documents/projects/TractSeg/dataset/camcan"      # name of folder that contains all the preprocessed subjects (each subject has its own folder with the name of the subjectID)
    FEATURES_FILENAME = "mrtrix_peaks"  # filename of nifti file (*.nii.gz) without file ending; mrtrix CSD peaks; shape: [x,y,z,9]; one file for each subject
    LABELS_FILENAME = 'corrected_tract_masks'
    DATASET = 'camcan'
    CLASSES = 'camcan_all_classes'
    NR_OF_CLASSES = 61

    SPATIAL_TRANSFORM = 'SpatialTransformPeaks'

    TRAIN = True
    NUM_EPOCHS = 250
    LEARNING_RATE = 0.001
    LR_SCHEDULE = True
    LR_SCHEDULE_MODE = "min"  # min | max
    LR_SCHEDULE_PATIENCE = 20
    OPTIMIZER = 'Adamax'
    LOSS_FUNCTION = 'soft_sample_dice'
    BATCH_SIZE = 8
    BATCH_NORM = False
    WEIGHT_DECAY = 0
    USE_DROPOUT = False
    DROPOUT_SAMPLING = False

    LOAD_WEIGHTS = False
    # WEIGHTS_PATH = join(C.EXP_PATH, "My_experiment/best_weights_ep64.npz")
    WEIGHTS_PATH = ""  # if empty string: autoloading the best_weights in get_best_weights_path()

    CV_FOLD = 0

    TEST = True
