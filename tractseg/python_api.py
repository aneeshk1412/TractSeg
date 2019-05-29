#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter("ignore", UserWarning)    #hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)    #hide h5py warnings
import importlib
import time
import os
from os.path import join
import numpy as np

from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs.system_config import get_config_name
from tractseg.libs import exp_utils
from tractseg.libs import utils
from tractseg.libs import dataset_utils
from tractseg.libs import direction_merger
from tractseg.libs import peak_utils
from tractseg.libs import img_utils
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.libs import trainer
from tractseg.models.base_model import BaseModel


def run_tractseg(data, output_type="tract_segmentation",
                 single_orientation=False, dropout_sampling=False, threshold=0.5,
                 bundle_specific_threshold=False, get_probs=False, peak_threshold=0.1,
                 postprocess=False, peak_regression_part="All", input_type="peaks",
                 blob_size_thr=50, nr_cpus=-1, verbose=False, manual_exp_name=None,
                 inference_batch_size=1, tract_definition="TractQuerier+", bedpostX_input=False,
                 tract_segmentations_path=None):
    """
    Run TractSeg

    Args:
        data: input peaks (4D numpy array with shape [x,y,z,9])
        output_type: TractSeg can segment not only bundles, but also the end regions of bundles.
            Moreover it can create Tract Orientation Maps (TOM).
            'tract_segmentation' [DEFAULT]: Segmentation of bundles (72 bundles).
            'endings_segmentation': Segmentation of bundle end regions (72 bundles).
            'TOM': Tract Orientation Maps (20 bundles).
        single_orientation: Do not run model 3 times along x/y/z orientation with subsequent mean fusion.
        dropout_sampling: Create uncertainty map by monte carlo dropout (https://arxiv.org/abs/1506.02142)
        threshold: Threshold for converting probability map to binary map
        bundle_specific_threshold: Set threshold to lower for some bundles which need more sensitivity (CA, CST, FX)
        get_probs: Output raw probability map instead of binary map
        peak_threshold: All peaks shorter than peak_threshold will be set to zero
        postprocess: Simple postprocessing of segmentations: Remove small blobs and fill holes
        peak_regression_part: Only relevant for output type 'TOM'. If set to 'All' (default) it will return all
            72 bundles. If set to 'Part1'-'Part4' it will only run for a subset of the bundles to reduce memory
            load.
        input_type: Always set to "peaks"
        blob_size_thr: If setting postprocess to True, all blobs having a smaller number of voxels than specified in
            this threshold will be removed.
        nr_cpus: Number of CPUs to use. -1 means all available CPUs.
        verbose: Show debugging infos
        manual_exp_name: Name of experiment if do not want to use pretrained model but your own one
        inference_batch_size: batch size (higher: a bit faster but needs more RAM)
        tract_definition: Select which tract definitions to use. 'TractQuerier+' defines tracts mainly by their
            cortical start and end region. 'AutoPTX' defines tracts mainly by ROIs in white matter.
        bedpostX_input: Input peaks are generated by bedpostX
        tract_segmentations_path: todo

    Returns:
        4D numpy array with the output of tractseg
        for tract_segmentation:     [x, y, z, nr_of_bundles]
        for endings_segmentation:   [x, y, z, 2*nr_of_bundles]
        for TOM:                    [x, y, z, 3*nr_of_bundles]
    """
    start_time = time.time()

    if manual_exp_name is None:
        config = get_config_name(input_type, output_type, dropout_sampling=dropout_sampling,
                                 tract_definition=tract_definition, bedpostX_input=bedpostX_input)
        Config = getattr(importlib.import_module("tractseg.experiments.pretrained_models." + config), "Config")()
    else:
        Config = exp_utils.load_config_from_txt(join(C.EXP_PATH,
                                                     exp_utils.get_manual_exp_name_peaks(manual_exp_name, "Part1"),
                                                     "Hyperparameters.txt"))

    Config = exp_utils.get_correct_labels_type(Config)
    Config.VERBOSE = verbose
    Config.TRAIN = False
    Config.TEST = False
    Config.SEGMENT = False
    Config.GET_PROBS = get_probs
    Config.LOAD_WEIGHTS = True
    Config.DROPOUT_SAMPLING = dropout_sampling
    Config.THRESHOLD = threshold
    Config.NR_CPUS = nr_cpus
    Config.INPUT_DIM = exp_utils.get_correct_input_dim(Config)

    if bundle_specific_threshold:
        Config.GET_PROBS = True

    if manual_exp_name is not None and Config.EXPERIMENT_TYPE != "peak_regression":
        Config.WEIGHTS_PATH = exp_utils.get_best_weights_path(join(C.EXP_PATH, manual_exp_name), True)
    else:
        if tract_definition == "TractQuerier+":
            if input_type == "peaks":
                if Config.EXPERIMENT_TYPE == "tract_segmentation" and Config.DROPOUT_SAMPLING:
                    Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_dropout_v2.npz")
                elif Config.EXPERIMENT_TYPE == "tract_segmentation":
                    if bedpostX_input:
                        Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "TractSeg_BXTensAg_best_weights_ep248.npz")
                    else:
                        Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_v2.npz")
                elif Config.EXPERIMENT_TYPE == "endings_segmentation":
                    Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_endings_segmentation_v3.npz")
                elif Config.EXPERIMENT_TYPE == "dm_regression":
                    Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_dm_regression_v1.npz")
            else:  # T1
                if Config.EXPERIMENT_TYPE == "tract_segmentation":
                    Config.WEIGHTS_PATH = join(C.NETWORK_DRIVE, "hcp_exp_nodes/x_Pretrained_TractSeg_Models",
                                               "TractSeg_T1_125mm_DAugAll", "best_weights_ep142.npz")
                elif Config.EXPERIMENT_TYPE == "endings_segmentation":
                    Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_endings_segmentation_v1.npz")
                elif Config.EXPERIMENT_TYPE == "peak_regression":
                    Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_peak_regression_v1.npz")
        else:  # AutoPTX
            if Config.EXPERIMENT_TYPE == "tract_segmentation":
                Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_aPTX_v1.npz")
            elif Config.EXPERIMENT_TYPE == "dm_regression":
                Config.WEIGHTS_PATH = join(C.WEIGHTS_DIR, "pretrained_weights_dm_regression_aPTX_v1.npz")
            else:
                raise ValueError("bundle_definition AutoPTX not supported in combination with this output type")
            #todo: remove when aPTX weights are loaded automatically
            if not os.path.exists(Config.WEIGHTS_PATH):
                raise FileNotFoundError("Could not find weights file: {}".format(Config.WEIGHTS_PATH))

    if Config.VERBOSE:
        print("Hyperparameters:")
        exp_utils.print_Configs(Config)

    data = np.nan_to_num(data)
    # brain_mask = img_utils.simple_brain_mask(data)
    # if Config.VERBOSE:
    #     nib.save(nib.Nifti1Image(brain_mask, np.eye(4)), "otsu_brain_mask_DEBUG.nii.gz")

    if input_type == "T1":
        data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    data, seg_None, bbox, original_shape = dataset_utils.crop_to_nonzero(data)
    data, transformation = dataset_utils.pad_and_scale_img_to_square_img(data, target_size=Config.INPUT_DIM[0])

    if Config.EXPERIMENT_TYPE == "tract_segmentation" or Config.EXPERIMENT_TYPE == "endings_segmentation" or \
            Config.EXPERIMENT_TYPE == "dm_regression":
        print("Loading weights from: {}".format(Config.WEIGHTS_PATH))
        Config.NR_OF_CLASSES = len(exp_utils.get_bundle_names(Config.CLASSES)[1:])
        utils.download_pretrained_weights(experiment_type=Config.EXPERIMENT_TYPE,
                                          dropout_sampling=Config.DROPOUT_SAMPLING)
        model = BaseModel(Config)
        if single_orientation:  # mainly needed for testing because of less RAM requirements
            data_loder_inference = DataLoaderInference(Config, data=data)
            if Config.DROPOUT_SAMPLING or Config.EXPERIMENT_TYPE == "dm_regression" or Config.GET_PROBS:
                seg, img_y = trainer.predict_img(Config, model, data_loder_inference, probs=True,
                                                 scale_to_world_shape=False, only_prediction=True,
                                                 batch_size=inference_batch_size)
            else:
                seg, img_y = trainer.predict_img(Config, model, data_loder_inference, probs=False,
                                                 scale_to_world_shape=False, only_prediction=True,
                                                 batch_size=inference_batch_size)
        else:
            seg_xyz, gt = direction_merger.get_seg_single_img_3_directions(Config, model, data=data,
                                                                           scale_to_world_shape=False,
                                                                           only_prediction=True,
                                                                           batch_size=inference_batch_size)
            if Config.DROPOUT_SAMPLING or Config.EXPERIMENT_TYPE == "dm_regression" or Config.GET_PROBS:
                seg = direction_merger.mean_fusion(Config.THRESHOLD, seg_xyz, probs=True)
            else:
                seg = direction_merger.mean_fusion(Config.THRESHOLD, seg_xyz, probs=False)

    elif Config.EXPERIMENT_TYPE == "peak_regression":
        weights = {
            "Part1": "pretrained_weights_peak_regression_part1_v1.npz",
            "Part2": "pretrained_weights_peak_regression_part2_v1.npz",
            "Part3": "pretrained_weights_peak_regression_part3_v1.npz",
            "Part4": "pretrained_weights_peak_regression_part4_v1.npz",
        }
        if peak_regression_part == "All":
            parts = ["Part1", "Part2", "Part3", "Part4"]
            seg_all = np.zeros((data.shape[0], data.shape[1], data.shape[2], Config.NR_OF_CLASSES * 3))
        else:
            parts = [peak_regression_part]
            Config.CLASSES = "All_" + peak_regression_part
            Config.NR_OF_CLASSES = 3 * len(exp_utils.get_bundle_names(Config.CLASSES)[1:])

        for idx, part in enumerate(parts):
            if manual_exp_name is not None:
                manual_exp_name_peaks = exp_utils.get_manual_exp_name_peaks(manual_exp_name, part)
                Config.WEIGHTS_PATH = exp_utils.get_best_weights_path(
                    join(C.EXP_PATH, manual_exp_name_peaks), True)
            else:
                Config.WEIGHTS_PATH = join(C.TRACT_SEG_HOME, weights[part])
            print("Loading weights from: {}".format(Config.WEIGHTS_PATH))
            Config.CLASSES = "All_" + part
            Config.NR_OF_CLASSES = 3 * len(exp_utils.get_bundle_names(Config.CLASSES)[1:])
            utils.download_pretrained_weights(experiment_type=Config.EXPERIMENT_TYPE,
                                              dropout_sampling=Config.DROPOUT_SAMPLING, part=part)
            model = BaseModel(Config)

            if single_orientation:
                data_loder_inference = DataLoaderInference(Config, data=data)
                seg, img_y = trainer.predict_img(Config, model, data_loder_inference, probs=True,
                                                 scale_to_world_shape=False, only_prediction=True,
                                                 batch_size=inference_batch_size)
            else:
                # 3 dir for Peaks -> bad results
                seg_xyz, img_y = direction_merger.get_seg_single_img_3_directions(Config, model, data=data,
                                                                                  scale_to_world_shape=False,
                                                                                  only_prediction=True,
                                                                                  batch_size=inference_batch_size)
                seg = direction_merger.mean_fusion_peaks(seg_xyz)

            if peak_regression_part == "All":
                seg_all[:, :, :, (idx*Config.NR_OF_CLASSES) : (idx*Config.NR_OF_CLASSES+Config.NR_OF_CLASSES)] = seg

        if peak_regression_part == "All":
            Config.CLASSES = "All"
            Config.NR_OF_CLASSES = 3 * len(exp_utils.get_bundle_names(Config.CLASSES)[1:])
            seg = seg_all

        #quite fast
        # if bundle_specific_threshold:
        #     seg = peak_utils.remove_small_peaks_bundle_specific(seg, exp_utils.get_bundle_names(Config.CLASSES)[1:],
        #                                                        len_thr=0.3)
        # else:
        #     seg = peak_utils.remove_small_peaks(seg, len_thr=peak_threshold)

    if Config.EXPERIMENT_TYPE == "tract_segmentation" and bundle_specific_threshold:
        seg = img_utils.probs_to_binary_bundle_specific(seg, exp_utils.get_bundle_names(Config.CLASSES)[1:])

    #remove following two lines to keep super resolution
    seg = dataset_utils.cut_and_scale_img_back_to_original_img(seg, transformation)  # quite slow
    seg = dataset_utils.add_original_zero_padding_again(seg, bbox, original_shape, Config.NR_OF_CLASSES)  # quite slow

    if Config.EXPERIMENT_TYPE == "peak_regression":
        seg = peak_utils.mask_and_normalize_peaks(seg, tract_segmentations_path,
                                                  exp_utils.get_bundle_names(Config.CLASSES)[1:])

    if postprocess:
        seg = img_utils.postprocess_segmentations(seg, blob_thr=blob_size_thr, hole_closing=2)

    exp_utils.print_verbose(Config, "Took {}s".format(round(time.time() - start_time, 2)))
    return seg

