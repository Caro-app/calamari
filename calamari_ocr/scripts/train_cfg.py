import argparse
import os

from calamari_ocr import __version__
from calamari_ocr.utils import glob_all, split_all_ext, keep_files_with_same_file_name
from calamari_ocr.ocr.datasets import create_dataset, DataSetType, DataSetMode
from calamari_ocr.ocr.augmentation.data_augmenter import SimpleDataAugmenter
from calamari_ocr.ocr import Trainer
from calamari_ocr.ocr.text_processing import \
    default_text_normalizer_params, default_text_regularizer_params

from calamari_ocr.proto import CheckpointParams, DataPreprocessorParams, TextProcessorParams, \
    network_params_from_definition_string, NetworkParams, TextGeneratorParameters, LineGeneratorParameters

from google.protobuf import json_format

from calamari_ocr.proto.config import get_cfg
from yacs.config import CfgNode


def create_train_dataset(cfg: CfgNode, dataset_args=None):
    gt_extension = cfg.DATASET.TRAIN.GT_EXTENSION if cfg.DATASET.TRAIN.GT_EXTENSION is not False else DataSetType.gt_extension(cfg.DATASET.TRAIN.TYPE)

    # Training dataset
    print("Resolving input files")
    input_image_files = sorted(glob_all(cfg.DATASET.TRAIN.PATH))
    if not cfg.DATASET.TRAIN.TEXT_FILES:
        if gt_extension:
            gt_txt_files = [split_all_ext(f)[0] + gt_extension for f in input_image_files]
        else:
            gt_txt_files = [None] * len(input_image_files)
    else:
        gt_txt_files = sorted(glob_all(cfg.DATASET.TRAIN.TEXT_FILES))
        input_image_files, gt_txt_files = keep_files_with_same_file_name(input_image_files, gt_txt_files)
        for img, gt in zip(input_image_files, gt_txt_files):
            if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                raise Exception("Expected identical basenames of file: {} and {}".format(img, gt))

    if len(set(gt_txt_files)) != len(gt_txt_files):
        raise Exception("Some image are occurring more than once in the data set.")

    dataset = create_dataset(
        cfg.DATASET.TRAIN.TYPE,
        DataSetMode.TRAIN,
        images=input_image_files,
        texts=gt_txt_files,
        skip_invalid=not cfg.DATALOADER.NO_SKIP_INVALID_GT,
        args=dataset_args if dataset_args else {},
    )
    print("Found {} files in the dataset".format(len(dataset)))
    return dataset


def run(cfg: CfgNode):

    # check if loading a json file
    if len(cfg.DATASET.TRAIN.PATH) == 1 and cfg.DATASET.TRAIN.PATH[0].endswith("json"):
        import json
        with open(cfg.DATASET.TRAIN.PATH[0], 'r') as f:
            json_args = json.load(f)
            for key, value in json_args.items():
                if key == 'dataset' or key == 'validation_dataset':
                    setattr(cfg, key, DataSetType.from_string(value))
                else:
                    setattr(cfg, key, value)

    # parse whitelist
    whitelist = cfg.MODEL.CODEX.WHITELIST
    if len(whitelist) == 1:
        whitelist = list(whitelist[0])

    whitelist_files = glob_all(cfg.MODEL.CODEX.WHITELIST_FILES)
    for f in whitelist_files:
        with open(f) as txt:
            whitelist += list(txt.read())

    if cfg.DATASET.TRAIN.GT_EXTENSION is False:
        cfg.DATASET.TRAIN.GT_EXTENSION = DataSetType.gt_extension(cfg.DATASET.TRAIN.TYPE)

    if cfg.DATASET.VALID.GT_EXTENSION is False:
        cfg.DATASET.VALID.GT_EXTENSION = DataSetType.gt_extension(cfg.DATASET.VALID.TYPE)


    text_generator_params = TextGeneratorParameters()

    line_generator_params = LineGeneratorParameters()

    dataset_args = {
        'line_generator_params': line_generator_params,
        'text_generator_params': text_generator_params,
        'pad': None,
        'text_index': 0,
    }

    # Training dataset
    dataset = create_train_dataset(cfg, dataset_args)

    # Validation dataset
    if cfg.DATASET.VALID.PATH:
        print("Resolving validation files")
        validation_image_files = glob_all(cfg.DATASET.VALID.PATH)
        if not cfg.DATASET.VALID.TEXT_FILES:
            val_txt_files = [split_all_ext(f)[0] + cfg.DATASET.VALID.GT_EXTENSION for f in validation_image_files]
        else:
            val_txt_files = sorted(glob_all(cfg.DATASET.VALID.TEXT_FILES))
            validation_image_files, val_txt_files = keep_files_with_same_file_name(validation_image_files, val_txt_files)
            for img, gt in zip(validation_image_files, val_txt_files):
                if split_all_ext(os.path.basename(img))[0] != split_all_ext(os.path.basename(gt))[0]:
                    raise Exception("Expected identical basenames of validation file: {} and {}".format(img, gt))

        if len(set(val_txt_files)) != len(val_txt_files):
            raise Exception("Some validation images are occurring more than once in the data set.")

        validation_dataset = create_dataset(
            cfg.DATASET.VALID.TYPE,
            DataSetMode.TRAIN,
            images=validation_image_files,
            texts=val_txt_files,
            skip_invalid=not cfg.DATALOADER.NO_SKIP_INVALID_GT,
            args=dataset_args,
        )
        print("Found {} files in the validation dataset".format(len(validation_dataset)))
    else:
        validation_dataset = None

    params = CheckpointParams()

    params.max_iters = cfg.SOLVER.MAX_ITER
    params.stats_size = cfg.STATS_SIZE
    params.batch_size = cfg.SOLVER.BATCH_SIZE
    params.checkpoint_frequency = cfg.SOLVER.CHECKPOINT_FREQ if cfg.SOLVER.CHECKPOINT_FREQ >= 0 else cfg.SOLVER.EARLY_STOPPING_FREQ
    params.output_dir = cfg.OUTPUT_DIR
    params.output_model_prefix = cfg.OUTPUT_MODEL_PREFIX
    params.display = cfg.DISPLAY
    params.skip_invalid_gt = not cfg.DATALOADER.NO_SKIP_INVALID_GT
    params.processes = cfg.NUM_THREADS
    params.data_aug_retrain_on_original = not cfg.DATALOADER.ONLY_TRAIN_ON_AUGMENTED

    params.early_stopping_at_acc = cfg.SOLVER.EARLY_STOPPING_AT_ACC
    params.early_stopping_frequency = cfg.SOLVER.EARLY_STOPPING_FREQ
    params.early_stopping_nbest = cfg.SOLVER.EARLY_STOPPING_NBEST
    params.early_stopping_best_model_prefix = cfg.EARLY_STOPPING_BEST_MODEL_PREFIX
    params.early_stopping_best_model_output_dir = \
        cfg.EARLY_STOPPING_BEST_MODEL_OUTPUT_DIR if cfg.EARLY_STOPPING_BEST_MODEL_OUTPUT_DIR else cfg.OUTPUT_DIR

    if cfg.INPUT.DATA_PREPROCESSING is False or len(cfg.INPUT.DATA_PREPROCESSING) == 0:
        cfg.INPUT.DATA_PREPROCESSING = [DataPreprocessorParams.DEFAULT_NORMALIZER]

    params.model.data_preprocessor.type = DataPreprocessorParams.MULTI_NORMALIZER
    for preproc in cfg.INPUT.DATA_PREPROCESSING:
        pp = params.model.data_preprocessor.children.add()
        pp.type = DataPreprocessorParams.Type.Value(preproc) if isinstance(preproc, str) else preproc
        pp.line_height = cfg.INPUT.LINE_HEIGHT
        pp.pad = cfg.INPUT.PAD

    # Text pre processing (reading)
    params.model.text_preprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(params.model.text_preprocessor.children.add(), default=cfg.INPUT.TEXT_NORMALIZATION)
    default_text_regularizer_params(params.model.text_preprocessor.children.add(), groups=cfg.INPUT.TEXT_REGULARIZATION)
    strip_processor_params = params.model.text_preprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    # Text post processing (prediction)
    params.model.text_postprocessor.type = TextProcessorParams.MULTI_NORMALIZER
    default_text_normalizer_params(params.model.text_postprocessor.children.add(), default=cfg.INPUT.TEXT_NORMALIZATION)
    default_text_regularizer_params(params.model.text_postprocessor.children.add(), groups=cfg.INPUT.TEXT_REGULARIZATION)
    strip_processor_params = params.model.text_postprocessor.children.add()
    strip_processor_params.type = TextProcessorParams.STRIP_NORMALIZER

    if cfg.SEED > 0:
        params.model.network.backend.random_seed = cfg.SEED

    if cfg.INPUT.BIDI_DIR:
        # change bidirectional text direction if desired
        bidi_dir_to_enum = {"rtl": TextProcessorParams.BIDI_RTL, "ltr": TextProcessorParams.BIDI_LTR,
                            "auto": TextProcessorParams.BIDI_AUTO}

        bidi_processor_params = params.model.text_preprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = bidi_dir_to_enum[cfg.INPUT.BIDI_DIR]

        bidi_processor_params = params.model.text_postprocessor.children.add()
        bidi_processor_params.type = TextProcessorParams.BIDI_NORMALIZER
        bidi_processor_params.bidi_direction = TextProcessorParams.BIDI_AUTO

    params.model.line_height = cfg.INPUT.LINE_HEIGHT
    params.model.network.learning_rate = cfg.SOLVER.LR
    params.model.network.lr_decay = cfg.SOLVER.LR_DECAY
    params.model.network.lr_decay_freq = cfg.SOLVER.LR_DECAY_FREQ
    network_params_from_definition_string(cfg.MODEL.NETWORK, params.model.network)
    params.model.network.clipping_norm = cfg.SOLVER.GRADIENT_CLIPPING_NORM
    params.model.network.backend.num_inter_threads = 0
    params.model.network.backend.num_intra_threads = 0
    params.model.network.backend.shuffle_buffer_size = cfg.DATALOADER.SHUFFLE_BUFFER_SIZE

    if cfg.MODEL.WEIGHTS == "":
        weights = None
    else:
        weights = cfg.MODEL.WEIGHTS

    # create the actual trainer
    trainer = Trainer(params,
                      dataset,
                      validation_dataset=validation_dataset,
                      data_augmenter=SimpleDataAugmenter(),
                      n_augmentations=cfg.INPUT.N_AUGMENT,
                      weights=weights,
                      codec_whitelist=whitelist,
                      keep_loaded_codec=cfg.MODEL.CODEX.KEEP_LOADED_CODEC,
                      preload_training=not cfg.DATALOADER.TRAIN_ON_THE_FLY,
                      preload_validation=not cfg.DATALOADER.VALID_ON_THE_FLY,
                      )
    trainer.train(
        auto_compute_codec=not cfg.MODEL.CODEX.SEE_WHITELIST,
        progress_bar=not cfg.NO_PROGRESS_BAR
    )


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1]
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    run(cfg)
