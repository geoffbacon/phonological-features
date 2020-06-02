# Configuration file for neural language model

## Hyperparameters
local NAME = "lstm";
local SIZE = 50;
local HIDDEN = 200;

## Level options
local LEVEL = "word";
local IS_PHONEME = LEVEL == "phoneme";

## Language options
local LANGUAGE = "nl";
local DATA_PREFIX = if IS_PHONEME then "data/phoneme/wikipron/" else "data/word/wiki40b/";
local TRAIN_DATA_PATH = DATA_PREFIX + LANGUAGE + "/train.txt";
local VALID_DATA_PATH = DATA_PREFIX + LANGUAGE + "/validation.txt";

# Ground truth representations for phonemes
local USE_GROUND_TRUTH = SIZE == "groundTruth";
local CHAR_DIM = if USE_GROUND_TRUTH then 22 else SIZE;
local GROUND_TRUTH = "data/phoneme/features/" + LANGUAGE + "/features.txt";

# Other options
local USE_GPU = false;
local BATCH_SIZE = 16;
local NUM_EPOCHS = 20;

local CONTEXTUALIZER = {
    type: NAME,
    input_size: CHAR_DIM,
    hidden_size: HIDDEN,
    num_layers: 1,
    bidirectional: false
};

local READER = {
    type: "simple_language_modeling",
    tokenizer: {
        type: "word",
        word_splitter: {
            type: "just_spaces"
        },
        word_filter: {
            type: "pass_through"
        },
        word_stemmer: {
            type: "pass_through"
        }
    },
    token_indexers: {
        tokens: {
            type: "single_id",
            lowercase_tokens: false
        }
    }
};

## Static
{
    dataset_reader: READER,
    train_data_path: TRAIN_DATA_PATH,
    validation_data_path: VALID_DATA_PATH,
    iterator: {
        type: "basic",
        batch_size: BATCH_SIZE
    },
    trainer: {
        optimizer: {
            type: "adam"
        },
        num_epochs: NUM_EPOCHS,
        patience: 2,
        cuda_device: if USE_GPU then 2 else -1,
        shuffle: false,
        num_serialized_models_to_keep: -1,
    },
    model: {
        type: "language_model",
        text_field_embedder: {
            type: "basic",
            token_embedders: {
                tokens: {
                    type: "embedding",
                    embedding_dim: CHAR_DIM,
                    trainable: if USE_GROUND_TRUTH then false else true,
                    pretrained_file: if USE_GROUND_TRUTH then GROUND_TRUTH else ""
                }
            }
        },
        contextualizer: CONTEXTUALIZER,
        bidirectional: false
    },
}
