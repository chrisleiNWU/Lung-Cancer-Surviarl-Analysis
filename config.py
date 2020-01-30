##########################################################
############################UNet config###################
##########################################################

###########for training###############
#Network size
unet_input_size=(2356,1304,3);
target_size=(2356,1304);
#Training iteration
batch_size=10;
per_epoch=100000;
epo=1;
learning_rate=1e-5
#File path
# train_path='./data/train';
train_path='Your train dataset dir';
image_folder_name='train dataset image_folder_name'
label_folder_name='train dataset label_folder_name'
loss_log_dir='loss logion dir'
#########for testing##################
test_dir = "Your test dataset dir"
test_result_root_dir= "./test_result"
##########################################################
############################VGG16 config###################
##########################################################
VGG16_pretrained_model_path='.vgg.hdf5'
pooling='avg'#'avg' or 'max'