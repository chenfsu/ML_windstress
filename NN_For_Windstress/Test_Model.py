import os
from pandas import DataFrame
import pandas as pd
from constants.AI_params import *
from os.path import join

from timeseries_viz.scatter import TimeSeriesVisualizer
from config.MainConfig import get_usemodel_1d_config
from AI.models.modelSelector import select_1d_model

from sklearn.metrics import *

_model_col = 'TAU(MFT)'
_obs_col = 'TAU(obs)'
_nn_col = 'NN'

from sklearn import preprocessing

def main():
    config = get_usemodel_1d_config()
    test_1d_model(config)

def test_1d_model(config):
    """
    :param config:
    :return:
    """
    # *********** Reads the parameters ***********

    input_folder = config[ClassificationParams.input_folder]
    input_file = config[ClassificationParams.input_file]
    output_folder = config[ClassificationParams.output_folder]
    output_file_name = config[ClassificationParams.output_file_name]
    model_weights_file = config[ClassificationParams.model_weights_file]
    output_imgs_folder = config[ClassificationParams.output_imgs_folder]
    training_data_file = config[ClassificationParams.training_data_file]
    save_predictions = config[ClassificationParams.save_prediction]
    model_name = config[TrainingParams.config_name]

    # Builds the visualization object
    viz_obj = TimeSeriesVisualizer(disp_images=config[ClassificationParams.show_imgs],
                                     output_folder=output_imgs_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_1d_model(config)
    model = select_1d_model(config)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # Read data
    data_df_original = pd.read_csv(join(input_folder, input_file))
    data_df_training = pd.read_csv(join(input_folder, training_data_file))
    scaler = preprocessing.MinMaxScaler()
    # Read previous parameters and use those to initialize the scaler
    scaler.fit(data_df_original)

    data_norm_np = scaler.transform(data_df_original)
    data_norm_df = DataFrame(data_norm_np, columns=data_df_original.columns)

    print(F'Data shape: {data_norm_df.shape} Data axes {data_norm_df.axes}')

    # *********** Makes a dataframe to contain the DSC information **********
    metrics_params = config[ClassificationParams.metrics]
    metrics_dict = {met.name: met.value for met in metrics_params}

    # Check if the output fiels already exist, in thtat case read the df from it.
    if os.path.exists(join(output_imgs_folder, output_file_name)):
        data = pd.read_csv(join(output_imgs_folder, output_file_name), index_col=0)
    else:
        data_columns = list(metrics_dict.values())
        data = DataFrame(index=data_norm_df.index.values, columns=data_columns)

    # If we want to visualize the input data
    # ------------------- Making prediction -----------
    print('\t Making prediction....')
    x_eval = data_norm_df.loc[:, ['wind', 'sst', 'airT', 'Pressure']].values
    output_nn_all_norm = model.predict(x_eval, verbose=1)

    # TODO This is just a patch to get the proper unormalization, make the code yourself in the preproc library
    data_norm_df[_model_col] = output_nn_all_norm
    data_np_original_temp = scaler.inverse_transform(data_norm_df)
    data_norm_df = DataFrame(data_np_original_temp, columns=data_df_original.columns)
    output_nn_all_original = data_norm_df[_model_col].values

    mft = data_df_original.loc[:, [_model_col]].values
    obs = data_df_original.loc[:, [_obs_col]].values

    mse_mft = mean_squared_error(mft,obs)
    mse_nn = mean_squared_error(output_nn_all_original,obs)
    print(F'MSE MFT: {mse_mft}  NN: {mse_nn}')

    # *********** Iterates over each case *********
    # for i in range(1700):
    #     x_eval = data_norm_df.loc[i, ['wind', 'sst', 'airT', 'Pressure']].values
    #     output_nn_all = model.predict(np.reshape(x_eval, (1,4)), verbose=1)
    #     obs = data_norm_df.loc[i, [_obs_col]].values
    #     phym = data_norm_df.loc[i, [_model_col]].values
    #     print(F"Obs: {obs} MFT: {phym} NN: {output_nn_all}")

    data_df_original['NN'] = output_nn_all_original

    # TAU(MFT),wind,sst,airT,Pressure,TAU(obs)
    viz_obj.plot_multicolumns_from_df(data_df_original, column_names=[_model_col, 'NN'], x_axis=_obs_col,
                                      title=F'MSE MFT{mse_mft:.4f}, NN:{mse_nn:.4f}', legends=['MFT','NN'],
                                      file_name=F'{model_name}.png')

    if save_predictions:
        print('\t Saving Prediction...')
        # TODO at some point we will need to see if we can output more than one ctr
        data_df_original.to_csv(join(output_folder,output_file_name))

#             if compute_metrics:
#                 # Compute metrics
#                 print('\t Computing metrics....')
#                 for c_metric in metrics_params:  # Here we can add more metrics
#                     if c_metric == ClassificationMetrics.DSC_3D:
#                         metric_value = numpy_dice(output_nn_np, ctrs_np[0])
#                         data.loc[current_folder][c_metric.value] = metric_value
#                         print(F'\t\t ----- DSC: {metric_value:.3f} -----')
#                         if compute_original_resolution:
#                             metric_value = numpy_dice(output_nn_original_np,
#                                                       sitk.GetArrayViewFromImage(gt_ctr_original_itk))
#                             data.loc[current_folder][F'{ORIGINAL_TXT}_{c_metric.value}'] = metric_value
#                             print(F'\t\t ----- DSC: {metric_value:.3f} -----')
#
#                 # Saving the results every 10 steps
#                 if id_folder % 10 == 0:
#                     save_metrics_images(data, metric_names=list(metrics_dict.values()), viz_obj=viz_obj)
#                     data.to_csv(join(output_folder, output_file_name))
#
#             if save_imgs:
#                 print('\t Plotting images....')
#                 plot_intermediate_results(current_folder, data_columns, imgs_itk=imgs_itk[0],
#                                           gt_ctr_itk=ctrs_itk[0][0], nn_ctr_itk=output_nn_itk, data=data,
#                                           viz_obj=viz_obj, slices=save_imgs_slices, compute_metrics=compute_metrics)
#                 if compute_original_resolution:
#                     plot_intermediate_results(current_folder, data_columns, imgs_itk=[img_original_itk],
#                                               gt_ctr_itk=gt_ctr_original_itk,
#                                               nn_ctr_itk=output_nn_original_itk, data=data,
#                                               viz_obj=viz_obj, slices=save_imgs_slices, compute_metrics=compute_metrics,
#                                               prefix_name=ORIGINAL_TXT)
#         except Exception as e:
#             print("---------------------------- Failed {} error: {} ----------------".format(current_folder, e))
#         print(F'\t Done! Elapsed time {time.time()-t0:0.2f} seg')
#
#     if compute_metrics:
#         save_metrics_images(data, metric_names=list(metrics_dict.values()), viz_obj=viz_obj)
#         data.to_csv(join(output_folder, output_file_name))
#
#

if __name__ == '__main__':
    main()
