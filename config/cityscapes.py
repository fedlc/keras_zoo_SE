# Dataset
problem_type                 = 'segmentation'  # ['classification' | 'detection' | 'segmentation']
dataset_name                 = 'cityscapes'    # Dataset name
dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch

# Model
model_name                   = 'segnet_vgg'     # Model to use ['fcn8' | 'segnet_basic' | 'segnet_vgg' | 'resnetFCN' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']
freeze_layers_from           = None            # Freeze layers from 0 to this layer during training (Useful for finetunning) [None | 'base_model' | Layer_id]
show_model                   = True            # Show the architecture layers
load_imageNet                = False            # Load Imagenet weights and normalize following imagenet procedure
load_pretrained              = False           # Load a pretrained model for doing finetuning
weights_file                 = 'weights.hdf5'  # Training weight file name
# weights_file                 = '/datatmp/dvazquez/Experiments/cityscapes/resnetFCNfrozen/weights.hdf5'  # Training weight file name

## ----------------------------------------------------------
# Parameters

## To train: activate only train
## to predict/test: activate train and predict/test
train_model                  = False          # Train the model

# Single model
pred_model                   = False       # Predict using the model
test_model                   = True       # Test the predictions of the model

# SE model
SE_pred_model                = False       # Predict using already saved models from Snapshot Ensemble
SE_test_model                = False      # Test using predictions from Snapshot Ensemble

#dataset to use for prediction and test
validation_set               = True
test_set                     = False

# number of prediction images to save
nb_pred_images_to_save       = 20

SE_model_weights             = [0.3, 0.35, 0.35] #[0.0, 0.0, 0.2, 0.4, 0.4]



# Callback model check point (Single Model)
checkpoint_enabled           = False            # Enable the Callback
checkpoint_monitor           = 'val_jaccard'   # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
##changed to last
checkpoint_save_best_only    = False            # Save best (True) or last (False) model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 0               # Verbosity of the checkpoint

## Callback Snapshot Ensemble during training
## (activate both: learning rate schedule and snapshot model weights saving)
SE_enabled                   = True             # Enable the callbacks
SE_n_models                  = 3                # Number of snapshot models

# Training parameters
optimizer                    = 'sgd'          # Optimizer
learning_rate                = 0.0001          # Training learning rate
weight_decay                 = 0.  #  0.0005     # Weight decay or L2 parameter norm penalty
n_epochs                     = 210             # Number of epochs during training

##
# Debug
debug                        = False            # Use only few images for debuging
debug_images_train           = 20              # N images for training in debug mode (-1 means all)
debug_images_valid           = 10              # N images for validation in debug mode (-1 means all)
debug_images_test            = 10              # N images for testing in debug mode (-1 means all)
debug_n_epochs               = 4               # N of training epochs in debug mode

# Batch sizes
batch_size_train             = 5               # Batch size during training
batch_size_valid             = 10               # Batch size during validation
batch_size_test              = 10               # Batch size during testing
crop_size_train              = None #(512, 512)      # Crop size during training (Height, Width) or None
crop_size_valid              = None            # Crop size during validation
crop_size_test               = None            # Crop size during testing
resize_train                 = (360,480)  #(512, 1024)      # Resize the image during training (Height, Width) or None
resize_valid                 = (360,480)  #(512, 1024)      # Resize the image during validation
resize_test                  = (360,480)  #(512, 1024)      # Resize the image during testing

# Data shuffle
## To actually shuffle, we need seed=False (see class Iterator in keras.preprocessing.image), otherwise with fixed seed we get the same data permutation
shuffle_train                = True            # Whether to shuffle the training data
shuffle_valid                = False           # Whether to shuffle the validation data
shuffle_test                 = False           # Whether to shuffle the testing data
seed_train                   = 1924            # Random seed for the training shuffle
seed_valid                   = 1924            # Random seed for the validation shuffle
seed_test                    = 1924            # Random seed for the testing shuffle

max_q_size                   = 10              # Maximum size for the data generator queue
workers                      = 5               # Maximum number of processes to spin up when using process based threading

## ------------ MORE CALLBACKS ------------

# Callback save results
save_results_enabled         = True            # Enable the Callback
save_results_nsamples        = 5               # Number of samples to save
save_results_batch_size      = 5               # Size of the batch
save_results_n_legend_rows   = 1               # Number of rows when showwing the legend

# Callback early stoping
earlyStopping_enabled        = False            # Enable the Callback
earlyStopping_monitor        = 'val_jaccard'   # Metric to monitor
earlyStopping_mode           = 'max'           # Mode ['max' | 'min']
earlyStopping_patience       = 100              # Max patience for the early stopping
earlyStopping_verbose        = 0               # Verbosity of the early stopping

# Callback plot
plotHist_enabled             = True            # Enable the Callback
plotHist_verbose             = 0               # Verbosity of the callback

# Callback learning rate scheduler
LRScheduler_enabled          = False            # Enable the Callback
LRScheduler_batch_epoch      = 'batch'         # Schedule the LR each 'batch' or 'epoch'
LRScheduler_type             = 'linear'          # Type of scheduler ['linear' | 'step' | 'square' | 'sqrt' | 'poly']
LRScheduler_M                = 75000           # Number of iterations/epochs expected until convergence
LRScheduler_decay            = 0.1             # Decay for 'step' method
LRScheduler_S                = 10000           # Step for the 'step' method
LRScheduler_power            = 0.9             # Power for te poly method

# Callback TensorBoard
TensorBoard_enabled          = False           # Enable the Callback
TensorBoard_logs_folder      = None            # Logs folder. If None it would make /home/youruser/TensorBoardLogs/. Either put a regular path.
TensorBoard_histogram_freq   = 1               # Frequency (in epochs) at which to compute activation histograms for the layers of the model. If set to 0, histograms won't be computed.
TensorBoard_write_graph      = True            # Whether to visualize the graph in Tensorboard. The log file can become quite large when write_graph is set to True.
TensorBoard_write_images     = False           # Whether to write model weights to visualize as image in Tensorboard.

## ------------ DATA AUGMENTATION AND PREPROCESSING ------------

# Data augmentation for training and normalization
#to change this part
norm_imageNet_preprocess           = False     # Normalize following imagenet procedure
norm_fit_dataset                   = False      # If True it recompute std and mean from images. Either it uses the std and mean set at the dataset config file
norm_rescale                       = 1/255.    # Scalar to divide and set range 0-1
norm_featurewise_center            = True      # Substract mean - dataset
norm_featurewise_std_normalization = False#True      # Divide std - dataset
norm_samplewise_center             = False     # Substract mean - sample
norm_samplewise_std_normalization  = False     # Divide std - sample
norm_gcn                           = False     # Global contrast normalization
norm_zca_whitening                 = False     # Apply ZCA whitening
cb_weights_method                  = None      # Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']

# Data augmentation for training
da_rotation_range                  = 0         # Rnd rotation degrees 0-180
da_width_shift_range               = 0.0       # Rnd horizontal shift
da_height_shift_range              = 0.0       # Rnd vertical shift
da_shear_range                     = 0.0       # Shear in radians
da_zoom_range                      = 0.0 ##0.5      # Zoom
da_channel_shift_range             = 0.        # Channecf.l shifts
da_fill_mode                       = 'constant'# Fill mode
da_cval                            = 0.        # Void image value
da_horizontal_flip                 = False##True      # Rnd horizontal flip
da_vertical_flip                   = False     # Rnd vertical flip
da_spline_warp                     = False     # Enable elastic deformation
da_warp_sigma                      = 10        # Elastic deformation sigma
da_warp_grid_size                  = 3         # Elastic deformation gridSize
da_save_to_dir                     = False     # Save the images for debuging



## ------------ CHECKS ------------

#if ( (pred_model or test_model) and (SE_pred_model or SE_test_model) ):
#    raise Exception('It is possible to predict/test a single model XOR an ensemble model')

if ( train_model and (checkpoint_enabled == SE_enabled) ):
    raise Exception('During a train, better save snapshot models XOR best/last single model, to avoid confusion')

if ( train_model and (pred_model or test_model) and not checkpoint_enabled ):
    raise Exception('When training single model, you need checkpoint_enabled')

if ( train_model and (SE_pred_model or SE_test_model) and not SE_enabled ):
    raise Exception('When training ensemble model, you need SE_enabled')

if ( LRScheduler_enabled and SE_enabled):
    raise Exception('LRScheduler_enabled and SE_enabled are not compatible')

eps = 0.00001
if ( (train_model and SE_pred_model) and
                                        (SE_n_models!=len(SE_model_weights) or not
                                        (sum(SE_model_weights)<1+eps and sum(SE_model_weights)>1-eps))):
    raise Exception('Model weights are incorrect')
