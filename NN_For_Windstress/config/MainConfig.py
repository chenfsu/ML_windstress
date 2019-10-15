from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os

from AI.metrics import *
from constants.AI_params import *
from img_viz.constants import *

# ----------------------------- UM -----------------------------------
_data_folder = '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/data/csv'  # Where the data is stored and where the preproc folder will be saved
_run_name = F'Relu_Relu'  # Name of the model, for training and classification
_output_folder = '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/output'  # Where to save the models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code

def append_model_params(cur_config):
    model_config = {
        ModelParams.MODEL: AiModels.ML_PERCEPTRON,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: False,
        ModelParams.INPUT_SIZE: 4,
        ModelParams.HIDDEN_LAYERS: 3,
        ModelParams.CELLS_PER_HIDDEN_LAYER: [8, 8, 8],
        ModelParams.NUMBER_OF_OUTPUT_CLASSES: 1,
    }
    return {**cur_config, **model_config}


def get_training_1d():
    cur_config = {
        TrainingParams.input_folder: _data_folder,
        TrainingParams.output_folder: F"{join(_output_folder,'Training')}",
        TrainingParams.cases: 'all', # This can be also a numpy array
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.file_name: 'SWS2forML_nowave.csv',
        TrainingParams.evaluation_metrics: [metrics.mean_squared_error],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: losses.mean_squared_error,  # Loss function to use for the learning
        TrainingParams.optimizer: Adam(),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 30,
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False
    }
    return append_model_params(cur_config)


def get_usemodel_1d_config():
    models_folder = '/home/olmozavala/Dropbox/MyProjects/COAPS/ML_windstress/output/Training/models'
    model_file = 'Relu_Relu_2019_10_14_18_31-262-0.00094.hdf5'
    cur_config = {
        ClassificationParams.training_data_file: join(_data_folder, "SWS2forML_nowave.csv"),
        ClassificationParams.input_folder: _data_folder,
        ClassificationParams.output_folder: F"{join(_output_folder, 'Results')}",
        ClassificationParams.model_weights_file: join(models_folder, model_file),
        ClassificationParams.output_file_name: F'Results_{_run_name}.csv',
        ClassificationParams.input_file: 'zFAST_hr.csv',
        ClassificationParams.output_imgs_folder: F"{join(_output_folder, 'Results')}",
        ClassificationParams.show_imgs: True,
        ClassificationParams.save_prediction: True,
        ClassificationParams.metrics: [ClassificationMetrics.MSE],
        TrainingParams.config_name: _run_name,
    }
    return append_model_params(cur_config)
