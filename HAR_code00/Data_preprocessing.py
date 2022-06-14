from locale import D_FMT
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from pandas import read_csv
from numpy import dstack
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
from pandas import read_csv
from numpy import dstack
import pickle

#Dataset Option
WISDM0_UCI1 = 0
Aug_Option =  2
name_of_augmentation_algorithm = ["Original","SMOTE","ADASYN","BorderlineSMOTE","RandomOverSampler"]

if WISDM0_UCI1 == 0:
    number_of_each_class = 11000
else:
    number_of_each_class = 3500
    
dataset_name_WISDM0_UCI1 = ["WISDM", 'UCI_HAR']
train0_test1 = ["train", "test"]
save_path = 'HAR_code/HAR_data/' + dataset_name_WISDM0_UCI1[WISDM0_UCI1]

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

# load data ============================================================
if(WISDM0_UCI1 == 0):
    Fs = 20
    frame_size = Fs*4 # 80
    hop_size = Fs*2 # 40
    
    file = open('HAR_code/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
    lines = file.readlines()
    
    processedList = []
    
    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            last = line[5].split(';')[0]
            last = last.strip()
            if last == '':
                print('break!')
                #break;
            else: #original has no else
                temp = [line[0], line[1], line[2], line[3], line[4], last]
                processedList.append(temp)
        except:
            print('Error at line number: ', i)
            
    columns = ['user', 'activity', 'time', 'x', 'y', 'z']
    data = pd.DataFrame(data = processedList, columns = columns)
    #data.head()
    
    activities = data['activity'].value_counts().index
    activities
    
    df = data.drop(['user', 'time'], axis = 1).copy()
    #df.head()
    
    data['activity'].value_counts()
    
    # improve data imbalance problem
    
    label = LabelEncoder()
    data['label'] = label.fit_transform(data['activity'])
    #data.head()
    
    X = data[['x', 'y', 'z']]
    y = data['label']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])
    scaled_X['label'] = y.values
    scaled_X.head()
    
    X, y = get_frames(scaled_X, frame_size, hop_size)
    X.shape, y.shape        
    
    class_name = label.classes_
    
else:
    def load_file(filepath):
        dataframe = read_csv(filepath, header=None, delim_whitespace=True)
        return dataframe.values
        
    def load_group(filenames, prefix=''):
        loaded = list()
        for name in filenames:
            data = load_file(prefix + name)
            loaded.append(data)
        loaded = dstack(loaded)
        return loaded
    
    def load_dataset_group(group, prefix=''):
        filepath = prefix + group + '/Inertial Signals/'
        # load all 9 files as a single array
        filenames = list()
        # total acceleration
        filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
        # body acceleration
        filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
        # body gyroscope
        filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
        # load input data
        X = load_group(filenames, filepath)
        # load class output
        y = load_file(prefix + group + '/y_'+group+'.txt')
        return X, y   
    
    def load_dataset(prefix=''):
        	# load all train
        trainX, trainy = load_dataset_group('train', prefix + 'HAR_code/UCI/UCI HAR Dataset/')
        print(trainX.shape, trainy.shape)
        # load all test
        testX, testy = load_dataset_group('test', prefix + 'HAR_code/UCI/UCI HAR Dataset/')
        print(testX.shape, testy.shape)
        # zero-offset class values
        trainy = trainy - 1
        testy = testy - 1

        print(trainX.shape, trainy.shape, testX.shape, testy.shape)
        return trainX, trainy, testX, testy
    
    if Aug_Option != 0:
        X, y, X_test, y_test = load_dataset()
        X=np.concatenate((X,X_test), axis=0)
        y=np.concatenate((y,y_test), axis=0)
    else:
        X_train, y_train, X_test, y_test = load_dataset()
        
    class_name = np.array(["Walking","Walking Upstairs","Walking Downstairs","Sitting","Standing","Laying"])

unique, counts = np.unique(y, return_counts=True)
print('Before augmentation', dict(zip(unique, counts)))

# Oversampling based augmentation ==============================================
if Aug_Option != 0:
    X_oneD = np.zeros((len(X),len(X[0])*len(X[0][0])))
    for window_idx in range(len(X)):    
        buf = np.array([])
        for timestep_idx in range(len(X[0])):
                buf = np.append(buf, X[window_idx][timestep_idx])
        X_oneD[window_idx] = buf
    
    print('Data augmentation option : {}'.format(name_of_augmentation_algorithm[Aug_Option]))
    strategy = {0:number_of_each_class, 1:number_of_each_class, 2:number_of_each_class, 3:number_of_each_class, 4:number_of_each_class, 5:number_of_each_class}
    #이곳 SMOTE대신 ADASYN, BorderlineSMOTE, RandomOverSampler
    if Aug_Option == 1:
        oversample = SMOTE(k_neighbors=10, sampling_strategy=strategy) # k_neighbors 수가 모델보다 데이터가 부족할 수 있음
    elif Aug_Option == 2:
        oversample = ADASYN(n_neighbors=10, sampling_strategy=strategy, n_jobs=-1) # k_neighbors 수가 모델보다 데이터가 부족할 수 있음
    elif Aug_Option == 3:
        oversample = BorderlineSMOTE(k_neighbors=10, sampling_strategy=strategy) # k_neighbors 수가 모델보다 데이터가 부족할 수 있음
    elif Aug_Option == 4:
        oversample = RandomOverSampler(sampling_strategy=strategy) 

        
    X_AUG, y_AUG = oversample.fit_resample(X_oneD, y)
    X_AUG = np.reshape(X_AUG,(len(X_AUG),len(X[0]),len(X[0][0])))

    X_train, X_test, y_train, y_test = train_test_split(X_AUG, y_AUG, test_size = 0.3, random_state = 0, stratify = y_AUG)
    print('X_train.shape : {}  X_test.shape : {}'.format(X_train.shape, X_test.shape))

    shape_of_data = [(len(X_train),len(X_train[0]),len(X_train[0][0])), (len(X_test),len(X_test[0]),len(X_test[0][0]))]
    
    with open(save_path + "_shape_of_data.p", 'wb') as f:
        for line in shape_of_data:
            pickle.dump(line, f)
    
    with open(save_path + "_X_train_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'wb') as f:
        for line in X_train:
            pickle.dump(line, f)
    with open(save_path + "_y_train_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'wb') as f:
        for line in y_train:
            pickle.dump(line, f)
            
    with open(save_path + "_X_test_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'wb') as f:
        for line in X_test:
            pickle.dump(line, f)
    with open(save_path + "_y_test_"+ name_of_augmentation_algorithm[Aug_Option] +".p", 'wb') as f:
        for line in y_test:
            pickle.dump(line, f)

print('finish')
