from constants.AI_params import *
import AI.models.modelBuilder1D as model_builder_1d
from tensorflow.keras.layers import Input

def select_1d_model(model_params):
    model_type = model_params[ModelParams.MODEL]
    # Makes a 1D-Encoder
    if model_type == AiModels.ML_PERCEPTRON:
        # Reading configuration
        batch_normalization = model_params[ModelParams.BATCH_NORMALIZATION]
        dropout = model_params[ModelParams.DROPOUT]
        input_size = model_params[ModelParams.INPUT_SIZE]
        number_hidden_layers = model_params[ModelParams.HIDDEN_LAYERS]
        cells_per_hidden_layer = model_params[ModelParams.CELLS_PER_HIDDEN_LAYER]
        output_layer_size= model_params[ModelParams.NUMBER_OF_OUTPUT_CLASSES]
        # Setting the proper inputs
        inputs = Input(shape=(input_size,))
        # Building the model
        model = model_builder_1d.single_multlayer_perceptron( inputs, number_hidden_layers,
                                                              cells_per_hidden_layer,
                                                              output_layer_size,
                                                              dropout=dropout,
                                                              batch_norm=batch_normalization)
    else:
        raise Exception(F"The specified model doesn't have a configuration: {model_type.value}")

    return model
