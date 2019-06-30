This file shows where to alter specfic configeration options.

* Job Configuration (train_celebA_lisa.sh)* 

- Change gpu config (shared for testing purposes)
- Change TFrecord file directory to TFrecord location
- Change working directory
- Change directory to train.py location

* config.py configuration * 

- Change result_dir to local directory on home
- Change data_dir to TFrecords dir on scratch
- Change cache_dir to local directroy home

* train.py configuration *

- Configure training set to be used for sgan (if 1: in train.py)
- Configure number of GPU's to use ((un)comment line 47-50)
