#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:PREPROCFOLDER"

SRC_PATH='/home/olmozavala/Dropbox/MyProjects/OZ_LIB/AI_Template'
MAIN_CONFIG="${SRC_PATH}/config"

echo '############################ Training ############################ '
python $SRC_PATH/Train_Time_Series.py
