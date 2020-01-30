import cv2
import scipy.misc as misc
import config as cfg
import random
from vgg16 import VGG16
from UNet.model import *
from LLC.llc import llc
import lifelines
from sklearn_lifelines.estimators_wrappers import CoxPHFitterModel
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from patsylearn import PatsyTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
unet_input_size = cfg.unet_input_size
model = unet(unet_input_size)
model.load_weights("./u_net_tumor_cell_point.hdf5", by_name=True)


target_size = cfg.target_size
ori_size = cfg.unet_input_size
flag_multi_class = False
as_gray = True
def random_sample_index(List,chosen_num):
    dataList_index=range(len(List))
    sampled_index=[]
    for i in range(chosen_num):
        randIndex=int(random.uniform(0,len(dataList_index)))
        sampled_index.append(dataList_index[randIndex])
        del (dataList_index[randIndex])
    return sampled_index
def getFiles(dir, suffix):  # Find the root directory, file suffix
    res = []
    for root, directory, files in os.walk(dir):  # =>current root, root directory, file under directory
        for filename in files:
            name, suf = os.path.splitext(filename) 
            if suf == suffix:
                res.append(os.path.join(root, filename)) 
    return res


if __name__ == '__main__':
    flag_multi_class = False
    as_gray = True
    num_class = 1
    test_dir = cfg.test_dir
    test_result_root_dir =cfg.test_result_root_dir
    if not os.path.exists(test_result_root_dir):
        os.mkdir(test_result_root_dir)
    image_num = len(getFiles(test_dir, '.png'))
    cells_lists = []
    for file in getFiles(test_dir, '.png'):  
        filename = str(os.path.split(file)[-1].split('.')[0])
        image_ori = misc.imread(file)
        img = image_ori
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)
        test_img = model.predict(img)
        test_result = np.array(test_img[0][:, :, 0], dtype=float)
        nms_cell_heatmaps=test_result>0.5 #nms
        for i in range(nms_cell_heatmaps.shape[0]):
            for j in range(nms_cell_heatmaps.shape[1]):
                if nms_cell_heatmaps[i,j]!=0:
                    left_corner_x=i-40
                    left_corner_x=left_corner_x if left_corner_x>0 else 0
                    left_corner_y = j - 40
                    left_corner_y = left_corner_y if left_corner_y > 0 else 0
                    right_corner_x = i+40
                    right_corner_x = right_corner_x if right_corner_x > 0 else 0
                    right_corner_y = j + 40
                    right_corner_y = right_corner_y if right_corner_y > 0 else 0
                    cells_lists.append(image_ori[left_corner_x:right_corner_x,left_corner_y:right_corner_y])
    feature_extraction_model=VGG16(include_top=False,classes=2,input_shape=(80,80,3),pooling=cfg.pooling)
    feature_extraction_model.load_weights(cfg.VGG16_pretrained_model_path,by_name=True)
    sampled_cell_Indexs=random_sample_index(cells_lists,2000)#random sample 2000 cell features
    feature_lists=[]
    for index in sampled_cell_Indexs:
        a_cell_feature=feature_extraction_model.predict(cells_lists[index])
        feature_lists.append(a_cell_feature)
    feature_lists=np.array(feature_lists)
    C=llc(feature_lists)# the codes
    f=np.sum(C,axis=1)# the single vector of the patient.
    #f=np.max(C,axis=1)#the single vector of the patient.
    data = lifelines.datasets.load_dd()

    # create sklearn pipeline
    coxph_surv_ppl = make_pipeline(PatsyTransformer('un_continent_name + regime + start_year -1', \
                                                    return_type='dataframe'),
                                   CoxPHFitterModel(duration_column='duration', event_col='observed'))

    # split data to train and test
    data_train, data_test = train_test_split(data)

    # fit CoxPH model
    coxph_surv_ppl.fit(data_train, y=data_train)
    # use pipeline to predict expected lifetime
    exp_lifetime = coxph_surv_ppl.predict(data_test[0:1])
    print ('expected lifetime: ' + str(exp_lifetime))

    # or you can extract the model from the pipeline to access more methods
    coxmodel = coxph_surv_ppl.named_steps['coxphfittermodel'].estimator
    coxmodel.print_summary()
                    
                    