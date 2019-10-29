from datetime import datetime

from config.MainConfig import get_training_1d

from inout.io_common import create_folder

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing

from constants.AI_params import *
import AI.trainingutils as utilsNN
from AI.models.modelSelector import select_1d_model

import tensorflow as tf
from tensorflow.keras.utils import plot_model

from os.path import join


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == '__main__':

    config = get_training_1d()

    input_folder = config[TrainingParams.input_folder]
    output_folder = config[TrainingParams.output_folder]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    file_name = config[TrainingParams.file_name]
    model_name_user = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]
    optimizer = config[TrainingParams.optimizer]

    batch_size = config[TrainingParams.batch_size]
    data_augmentation = config[TrainingParams.data_augmentation]

    nn_input_size = config[ModelParams.INPUT_SIZE]
    model_type = config[ModelParams.MODEL]

    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)


    # =============== Read data and preprocess ===============
    data_df_original = pd.read_csv(join(input_folder,file_name))
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data_df_original)
    data_norm_np = scaler.transform(data_df_original)
    data_norm_df = DataFrame(data_norm_np, columns=data_df_original.columns)

    print(F'Data shape: {data_norm_df.shape} Data axes {data_norm_df.axes}')


    tot_examples = data_norm_df.shape[0]
    rows_to_read = np.arange(tot_examples)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    print("Train examples (total:{}) :{}".format(len(train_ids), rows_to_read[train_ids]))
    print("Validation examples (total:{}) :{}:".format(len(val_ids), rows_to_read[val_ids]))
    print("Test examples (total:{}) :{}".format(len(test_ids), rows_to_read[test_ids]))

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{model_name_user}_{now}'

    # ******************* Selecting the model **********************
    model = select_1d_model(config)
    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, folders_to_read=rows_to_read,
                        train_idx=train_ids, val_idx=val_ids, test_idx=test_ids)

    print(F"Norm params: {scaler.get_params()}")
    file_name_normparams = join(parameters_folder, F'{model_name}.txt')
    # utilsNN.save_norm_params(file_name_normparams, NormParams.min_max, scaler)
    utilsNN.save_norm_params(file_name_normparams, 1, scaler)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    print("Compiling model ...")
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Training ...")
    # This part should be somehow separated, it will change for every project
    x_train = data_norm_df.loc[train_ids, ['wind', 'sst','airT','Pressure']].values
    y_train = data_norm_df.loc[train_ids, ['TAU(obs)']].values
    x_val = data_norm_df.loc[val_ids, ['wind', 'sst','airT','Pressure']].values
    y_val = data_norm_df.loc[val_ids, ['TAU(obs)']].values
    x_test = data_norm_df.loc[test_ids, ['wind', 'sst','airT','Pressure']].values
    y_test = data_norm_df.loc[test_ids, ['TAU(obs)']].values


    history=model.fit(x_train, y_train,
                      batch_size=1024,
                      epochs=epochs,
                      validation_data=(x_val, y_val),
                      shuffle=True,
                      callbacks=[logger, save_callback, stop_callback])

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    # Evaluate all the groups (train, validation, test)
    # Unormalize and plot
