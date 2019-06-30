## Metrics 
For the metric evaluation we chose to focus on the Frechet Inception Distance (FID) for it's efficieny and wide adoptation for GANs. 

The original implementation by Nvidia contained two main issues: The pre-trained model often was not available from the Drive and the model only worked with RGB images, while the CHESTXRAY GAN produces grayscale images. The first issue was solved by making the pre-trained model available offline; its location can be specified in the config.py. The second problem has been resolved by changing the following aspects in frechet_inception_distance.py in the metrics folder:
1. line 42 - Convert grayscale Numpy array of shape (?, 1, 1024, 1024) to RGB array of size (?, 1, 1024, 1024) by duplicating the grayscale values across all 3 RGB color channels.
2. line 65 - Unpack image tensor of shape (?, 1, 1024, 1024) to Numpy array of the same shape. Apply conversion from previous step. Convert back to tensor of shape (x, 3, 1024, 1024).

## Conditionals
1. dataset_tools.py
	- Altered the function create_chestxray() which generates TF_records from the CHESTXRAY dataset labels.
	- Labels are optional, if no label path is provided, only TF_records will be generated.
	- Each label will be stored as a binary feature vector.
	- Because of this, a labelmap can be exported (optionally) that includes the following information:
		- The exported labels.
		- The index at which each label occurs in the feature vector.
		- NOTE: the labemap is represented as a dictionary. This means that the labels are not ordered.
		        It is advised to export the labelmap when exporting new labels.
	- See help(create_chestxray) for details.

## Misc
1. Added export_video.py (source: util_scripts.py from https://github.com/sara-nl/progressive_growing_of_gans) which contains a function to export a video of the training progress (with statistics) along with other required functions.
	- Added effective resolution statistic on the video (line 66).
	- Configure run_id in the file and run using `python export_video.py`
2. Added generate_images.py which contains a script for exporting labeled (not for sGAN) images, from a given pGAN or sGAN.
	- Set network, output directory, image name and amount of generated images (adds index when exporting multiple images).
	- Offers two option for image generation:
		- Generate a given amount of images with the same features
		- Generate a given amount of images with every feature as onehot vector multiple times
3. Added some example graphs in misc/network_figures.ipynb
	
## Horovod (WIP)
NOTE: The Horovod implementation is not yet ready. There are still some issues regarding data distribution and the way the gradients are handled. All the changes for the current Horovod implementation have been included as a copy with the format <original_filename>_hvd.py.

Quick overview of the changed files and the corresponding changes:
1. train_hvd.py 
	- num_gpu variable changed to 1 for every configuration to allow parallelization to be executed by Horovod.
	- initialisation of Horovod in the main function.
2. training/training_loop_hvd 
	- Wrapping of Tensorflow optimizers with Horovod distrubuted optimizer (line 197&198)
	- Broadcasting global variables (line 236)
	- Preventing double prints and exports by limiting certain processes to only the first worker using hvd.rank() (line 167, 183&184, 213, 228, 278, 294)
3. dnlib/submission/submit_hvd.py
	- Add the current rank to the run_id to prevent run folder creation errors. (line 168)
4. dnlib/tflib/tfutil_hvd.py
	- Pinning the GPU using local rank (line 135).
	
Overview of changes yet to be made:
1. Changing the source code of Horovod to make it sum the gradients instead of averaging them. Then re-install Horovod.
2. Find the code where the data from the TFrecords is originally divided into minibatches and ensure different minibatches for each Horovod worker.
