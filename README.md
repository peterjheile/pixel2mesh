In order to run this code, all files and subdirectoriews in the materials folder must be moved back a directory. This should fix all library and module import issues.
I just put them all in to follow the submission guidelines. There are three core files that you should use to run the code and see results.

The first of which is by navigating to the main directory and running: tensorboard --logdir=runs in the command line. This will allow you to
See all of my logged results from each of my runs. There should be a few different runs loss results you can check out. I could not upload all of them due to the
github upload limits.

Unfortuntatly, every one of my model checkpoints were unable to be loaded to github, this is due to file size. But you can make your own. Move to the train.py file and 
scroll down the log_dir and save_dir. These are the folders that the losses for tensorboard and the model checkpoints will be saved. There are also a few hyperparemeters that can be changed, along with more in the train() function. I highly recommend using a gpu and installing the requirements.txt for comapaility. Dont worry about creating the folders, they will automatically be created when you run the file.
NOTE: This only works if you manage to retrieve access to the shapenet dataset. I have both of the datasets I used as sources in my report.


The last main file that I believe you should check out is the p2m_model.py file. It contains the cumulation of the entire foward pass of the network. That means the logic for unsampling blocks, mutating meshes, and the key logic for the P2M network.


