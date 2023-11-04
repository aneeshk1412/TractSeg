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
    NUM_EPOCHS = 50
    PRINT_FREQ = 1
