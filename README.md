# InPainting

## This is an unofficial implementation of a research paper on InPainting i.e. completing images based on the surrounding pixel values

#### create a saved_sessions/main_session directory in the home directory i.e. InPainting/saved_session/main_session. This stores the session files to resume learning from last save checkpoint 

#### To start training run the train.py file. Training can be stopped at any time and resumed from the same point as checkpoints are saved at every 10th iteration

#### To make changes in the graph or parameters like batch size etc. and restart learning from scratch, delete the files inside saved_sessions/main_session directory and run the train.py file

###### Note: For replicating the environment used while creating the code use the environment.yml file and run command 'conda env create -f environment.yml' after installing anaconda
