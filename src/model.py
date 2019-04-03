from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Input, concatenate, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras_applications.resnet import ResNet101
from keras import backend, layers, utils, models # required for ResNet101 model import


class Models:
    '''
    This class contains the implementation of both AlexNet and ResNet101 
    HyperFace models as well as the R-CNN model
    
    Methods implemented in this class are - 
    
    # def R_CNN(self)
    # def hyperFace_AlexNet(self)
    # def get_HyperFace_AlexNet_parametres(self)
    # def get_HyperFace_ResNet101_parametres(self)
    # def hyperFace_ResNet101(self)
    # def plot_models(self,
                    model_name,
                    show_shapes = True, 
                    show_layer_names = True,
                    rankdir = 'TB')
    # def train_RCNN(self, train_data, validation_data)
    # def train_HyperFace_AlexNet(self, train_data, validation_data)
    # def train_HyperFace_ResNet101(self)
    # def get_model_summary(self, model_name)
        
        
    '''
    def __init__(self, lr, epochs, batch_size):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
    def R_CNN(self):
        ''' 
        Input image size : (227, 227, 3) 
        Output : out_face(2) 
        '''
        
        inputs = Input(shape=(227, 227, 3), name='input_tensor')

        conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', name='conv1')(inputs)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
        pool1 = BatchNormalization(name = 'batch_norm_1')(pool1)
        
        conv2 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='conv2')(pool1)
        pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool2')(conv2)
        pool2 = BatchNormalization(name = 'batch_norm_2')(pool2)
        
        conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3')(pool2)
        conv3 = BatchNormalization(name = 'batch_norm_3')(conv3)
        
        conv4 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv4')(conv3)
        conv4 = BatchNormalization(name = 'batch_norm_4')(conv4)
        
        conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv5')(conv4)
        pool5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(conv5)
        pool5 = BatchNormalization(name = 'batch_norm_5')(pool5)
        
        flatten = Flatten(name='flatten')(pool5)

        fully_connected = Dense(4096, activation='relu', name='fully_connected')(flatten)

        face_detection = Dense(512, activation='relu', name='detection')(fully_connected)

        out_face = Dense(2, name='face_detection_output')(face_detection)

        model = Model(inputs = inputs, outputs = out_face)

        model.compile(optimizer = Adam(lr = self.lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def hyperFace_AlexNet(self):
        ''' 
        Input image size : (227, 227, 3) 
        Output: out_detection(2), 
                out_landmarks(42),
                out_visibility(21), 
                out_pose(3),
                out_gender(2) 
        '''
        input_layer = Input(shape = (227, 227, 3), name = 'input_layer')

        conv1 = Conv2D(96, (11,11), strides = 4, activation = 'relu', padding = 'valid', name = 'conv1')            (input_layer)
        max1 = MaxPool2D((3,3), strides = 2, padding = 'valid',name = 'max1')(conv1)
        max1 = BatchNormalization(name = 'batch_norm_1')(max1)

        conv1a = Conv2D(256, (4,4), strides = 4, activation = 'relu', padding = 'valid', name = 'conv1a')(max1)
        conv1a = BatchNormalization(name = 'batch_norm_1a')(conv1a)
    
        conv2 = Conv2D(256, (5,5), strides = 1, activation = 'relu', padding = 'same', name = 'conv2')(max1)
        max2 = MaxPool2D((3,3), strides = 2, padding = 'valid', name = 'max2')(conv2)
        max2 = BatchNormalization(name = 'batch_norm_2')(max2)
    
        conv3 = Conv2D(384, (3,3), strides = 1, activation = 'relu', padding = 'same', name = 'conv3')(max2)
        conv3 = BatchNormalization(name = 'batch_norm_3')(conv3)
        conv3a = Conv2D(256, (2,2), strides = 2, activation = 'relu', padding = 'valid', name = 'conv3a')(conv3)
        conv3a = BatchNormalization(name = 'batch_norm_3a')(conv3a)

        conv4 = Conv2D(384, (3,3), strides = 1, activation = 'relu', padding = 'same', name = 'conv4')(conv3)
        conv4 = BatchNormalization(name = 'batch_norm_4')(conv4)
        conv5 = Conv2D(256, (3,3), strides = 1, activation = 'relu', padding = 'same', name = 'conv5')(conv4)
        pool5 = MaxPool2D((3,3), strides = 2, padding = 'valid', name = 'pool5')(conv5)
        pool5 = BatchNormalization(name = 'batch_norm_5')(pool5)

        concat = concatenate([conv1a, conv3a, pool5], name = 'concat')
        concat = BatchNormalization(name = 'batch_norm_concat')(concat)

        conv_all = Conv2D(192, (1,1), strides = 1, activation = 'relu', padding = 'valid', name = 'conv_all')(concat)
        flatten = Flatten(name = 'flatten')(conv_all)

        fc_full = Dense(3072, activation = 'relu', name = 'fc_full')(flatten)

        fc_detection = Dense(512, activation = 'relu', name = 'fc_detection')(fc_full)
        fc_landmarks = Dense(512, activation = 'relu', name = 'fc_landmarks')(fc_full)
        fc_visibility = Dense(512, activation = 'relu', name = 'fc_visibility')(fc_full)
        fc_pose = Dense(512, activation = 'relu', name = 'fc_pose')(fc_full)
        fc_gender = Dense(512, activation = 'relu', name = 'fc_gender')(fc_full)

        out_detection = Dense(1, activation = 'softmax', name = 'out_detection')(fc_detection)
        out_landmarks = Dense(42, activation = 'softmax', name = 'out_landmarks')(fc_landmarks)
        out_visibility = Dense(21, activation = 'sigmoid', name = 'out_visibility')(fc_visibility)
        out_pose = Dense(3, activation = 'softmax', name = 'out_pose')(fc_pose)
        out_gender = Dense(1, activation = 'softmax', name = 'out_gender')(fc_gender)

        model = Model(inputs = input_layer, outputs = [out_detection, out_landmarks, out_visibility, out_pose, out_gender])
        
        losses, loss_weights, optimizer, callbacks = self.getHyperFace_AlexNet_parametres()
        
        model.compile(optimizer = optimizer, loss = losses, loss_weights = loss_weights, metrics = ['accuracy'], callbacks = callbacks)
        
        return model
    
    def get_HyperFace_AlexNet_parametres(self):
        '''
        Returns losses, loss_weights, and optimizer used in HyperFace_Alexnet 
        '''
        
        losses = {
                
                "out_detection" : "sparse_categorical_crossentropy",
                "out_landmarks" : "mean_squared_error",
                "out_visibility" : "mean_squared_error",
                "out_pose" : "mean_squared_error",
                "out_gender" : "binary_crossentropy"
                }

        loss_weights = {
                "out_detection" : 1.0,
                "out_landmarks" : 5.0,
                "out_visibility" : 0.5,
                "out_pose" : 5.0,
                "out_gender" : 2.0
                }

        optimizer = RMSprop(
                lr = self.lr,
                rho = 0.9,
                epsilon = None,
                decay = 0.0
                )
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor = 0.2,
                                      patience = 3,
                                      min_lr = 0.00001)
        
        callbacks = [reduce_lr]
        
        return losses, loss_weights, optimizer, callbacks
    
    def hyperFace_ResNet101(self):
        '''
        This function returns ResNet101 model with conv layers trained on 
        ImageNet dataset.
        
        keras_applications.resnet - change this function in future when 
        ResNet101 is implemented in a future version of keras.
        
        Input image size : (227, 227, 3) 
        Output: out_detection(2), 
                out_landmarks(42),
                out_visibility(21), 
                out_pose(3),
                out_gender(2) 
        
        '''
        
        
        
        # DO the implementation
        
        model = ResNet101(include_top = False,
                          weights='imagenet',
                          input_tensor = None,
                          input_shape = (227,227, 3),
                          pooling = 'avg',
                          backend = backend,
                          layers = layers,
                          models = models,
                          utils = utils
                          )
        
        
        return model
    
    def get_HyperFace_ResNet101_parametres(self):
        '''
        Returns losses, loss_weights, and optimizer used in HyperFace_ResNet101 
        '''
        
        losses = {
                
                "out_detection" : "sparse_categorical_crossentropy",
                "out_landmarks" : "mean_squared_error",
                "out_visibility" : "mean_squared_error",
                "out_pose" : "mean_squared_error",
                "out_gender" : "binary_crossentropy"
                }

        loss_weights = {
                "out_detection" : 1.0,
                "out_landmarks" : 1.0,
                "out_visibility" : 1.0,
                "out_pose" : 1.0,
                "out_gender" : 1.0
                }

        optimizer = RMSprop(
                lr = self.lr,
                rho = 0.9,
                epsilon = None,
                decay = 0.0
                )
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor = 0.2,
                                      patience = 3,
                                      min_lr = 0.00001)
        
        callbacks = [reduce_lr]
        
        return losses, loss_weights, optimizer, callbacks
    
    
        
        
        
    def plot_models(self,
                    model_name,
                    show_shapes = True, 
                    show_layer_names = True,
                    rankdir = 'TB'):
        '''
        Uses keras.utils.plot_model to plot a model
        '''
        
        if model_name == 'R_CNN':
            model = self.R_CNN()
        elif model_name == 'HyperFace_AlexNet':
            model = self.hyperFace_AlexNet()
        elif model_name == 'HyperFace_ResNet101':
            model = self.hyperFace_ResNet101()
        else:
            print("Please choose from the following options :")
            print("1) R_CNN")
            print("2) HyperFace_AlexNet")
            print("3) HyperFace_ResNet101")
        
        return utils.plot_model(model, 
                   to_file='model.png',
                   show_shapes = True,
                   show_layer_names = True,
                   rankdir = 'TB'
                   )
        
   
    def train_RCNN(self, train_data, validation_data):
        '''
        Trains the model on the train_data and validates it on validation_data
        '''
        print("Processing train and validation data...")
        x_train, y_train = train_data
        x_val, y_val = validation_data
        
        
        print("Training...")
        
        model = self.R_CNN()
        
        model.fit(x_train,
                  y_train,
                  validation_data = validation_data,
                  batch_size = self.batch_size,
                  epochs = self.epochs,
                  verbose = 1)
        
        print("Finished training")
        
        return
    
    
    def train_HyperFace_AlexNet(self, train_data, validation_data):
        '''
        Trains the model on the train_data and validates it on validation_data
        '''

        print("Processing train and validation data...")
        x_train, y_train_landmarks, y_train_visibility, y_train_pose, y_train_gender = train_data
        x_test, y_test_landmarks, y_test_visibility, y_test_pose, y_test_gender = validation_data
        
        x, y, z, callbacks = self.get_HyperFace_ResNet101_parametres

        print("Training...")
        
        model = self.hyperFace_AlexNet()
        
        model.fit(x_train,
                  {
                   "out_face" : y_train_face,
                   "out_landmarks" : y_train_landmarks,
                   "out_visibility" : y_train_visibility,
                   "out_pose" : y_train_pose,
                   "out_gender" : y_train_gender
                   },
                   validation_data = (x_test, 
                                      { 
                                        "out_face" : y_test_face
                                        "out_landmarks" : y_test_landmarks,
                                        "out_visibility" : y_test_visibility, 
                                        "out_pose" : y_test_pose, 
                                        "out_gender" : y_test_gender}
                                        ),
                   epochs = self.epochs,
                   verbose = 1, 
                   batch_size = self.batch_size, 
                   callbacks = callbacks
                   )
         
        print("Finished training")
         
        return
        
    def train_HyperFace_ResNet101(self, train_data, validation_data):
        '''
        Trains the model on the train_data and validates it on validation_data
        '''

        print("Processing train and validation data...")
        x_train, y_train_face, y_train_landmarks, y_train_visibility, y_train_pose, y_train_gender = train_data
        x_test, y_test_face, y_test_landmarks, y_test_visibility, y_test_pose, y_test_gender = validation_data
        
        x, y, z, callbacks = self.getHyperFace_AlexNet_parametres()

        print("Training...")
        
        model = self.hyperFace_ResNet101
        
        model.fit(x_train,
                  {
                   "out_face" : y_train_face,
                   "out_landmarks" : y_train_landmarks,
                   "out_visibility" : y_train_visibility,
                   "out_pose" : y_train_pose,
                   "out_gender" : y_train_gender
                   },
                   validation_data = (x_test, 
                                      { 
                                        "out_face" : y_test_face,
                                        "out_landmarks" : y_test_landmarks,
                                        "out_visibility" : y_test_visibility, 
                                        "out_pose" : y_test_pose, 
                                        "out_gender" : y_test_gender}
                                        ),
                   epochs = self.epochs,
                   verbose = 1, 
                   batch_size = self.batch_size, 
                   callbacks = callbacks
                   )
         
        print("Finished training")
         
        return
    
    
    def get_model_summary(self, model_name):
        '''
        Prints the selected model summary
        '''
        if model_name == 'R_CNN':
            model = self.R_CNN()
        elif model_name == 'HyperFace_AlexNet':
            model = self.hyperFace_AlexNet()
        elif model_name == 'HyperFace_ResNet101':
            model = self.hyperFace_ResNet101()
        else:
            print("Please choose from the following options :")
            print("1) R_CNN")
            print("2) HyperFace_AlexNet")
            print("3) HyperFace_ResNet101")
                       
        return model.summary()
        
    
        
        
        
        
    
