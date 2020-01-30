from keras.models import load_model
from UNet.model import *
from UNet.data import *
import keras.backend as K
import config as cfg
from keras.callbacks import LearningRateScheduler
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_save(self):
        path=loss_log_dir
        if not os.path.exists(path):os.makedirs(path)
        f=open(os.path.join(path,'result.txt'),'w')
        f.write('epoch\ttrain_loss\tval_loss\n')
        for i in range(len(self.losses['epoch'])):
            loss=self.losses['epoch'][i]
            val_loss=self.val_loss['epoch'][i]
            f.write(str(i)+'\t'+'%.3f'%loss+'\t%.3f'%val_loss+'\n')
        f.close()



#Network size
unet_input_size=cfg.unet_input_size
target_size=cfg.target_size
#Training iteration
batch_size=cfg.target_size
per_epoch=cfg.per_epoch
epo=cfg.epo
#File path
# train_path='./data/train';
train_path=cfg.train_path
image_folder_name=cfg.image_folder_name
label_folder_name=cfg.label_folder_name
loss_log_dir=cfg.loss_log_dir
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='constant',
                    )



myGene = trainGenerator(batch_size,train_path,image_folder_name,label_folder_name,target_size=target_size,aug_dict=data_gen_args,num_class=2)

model = unet(unet_input_size)

model_checkpoint = ModelCheckpoint('u_net_tumor_cell_point.hdf5',monitor='loss',verbose=1,period=10)

model.fit_generator(myGene,steps_per_epoch=per_epoch,epochs=epo,callbacks=[model_checkpoint])


