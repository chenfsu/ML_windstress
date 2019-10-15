Install
===========

In order to run the code its best to have [Anaconda](https://www.anaconda.com) installed. Then we can install
the required packages with `conda install --file requirements.txt`. 


Run
===========

The *training* and *testing* is configured at the `config/MainConfig.py` file. This file is not sync on the
repository so you need to create one for your paths. Just copy/paste the example file in the `config` folder. 

Training
---------
To perform the training execute `python Train_Time_Series.py`


Testing
---------
To test an specific model execute `python Test_Model.py`
