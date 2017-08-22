# import os
import math
import time
import os
import numpy as np
from keras.engine.training import GeneratorEnqueuer
from tools.save_images import save_img3
from keras import backend as K
# from tools.yolo_utils import *


# Keras dim orders
def channel_idx():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3


class Model():
    """
    Interface for normal (one net) models and adversarial models. Objects of
    classes derived from Model are returned by method make() of the
    Model_Factory class.
    """
    def train(self, train_gen, valid_gen, cb):
        pass

    def train2(self, load_data_func, cb):
        pass

    def predict(self, test_gen, tag='pred'):
        pass

    def test(self, test_gen):
        pass


# TODO: Better call it Regular_Model ?
class One_Net_Model(Model):
    """
    Wraper of regular models like FCN, SegNet etc consisting of a one Keras
    model. But not GANs, which are made of two networks and have a different
    training strategy. In this class we implement the train(), test() and
    predict() methods common to all of them.
    """
    def __init__(self, model, cf, optimizer):
        self.cf = cf
        self.optimizer = optimizer
        self.model = model

    # Train the model
    def train(self, train_gen, valid_gen, cb):
        if (self.cf.train_model):
            print('\n > Training the model...')
            hist = self.model.fit_generator(generator=train_gen,
                                            samples_per_epoch=self.cf.dataset.n_images_train,
                                            nb_epoch=self.cf.n_epochs,
                                            verbose=1,
                                            callbacks=cb,
                                            validation_data=valid_gen,
                                            nb_val_samples=self.cf.dataset.n_images_valid,
                                            class_weight=None,
                                            max_q_size=self.cf.max_q_size,
                                            nb_worker=self.cf.workers,
                                            pickle_safe=True)
            print('   Training finished.')

            return hist
        else:
            return None

    def train2(self, load_data_func, cb):
        if (self.cf.train_model):
            print('\n > Training the model...')
            # Load data
            (x_train, y_train), (x_test, y_test) = load_data_func(os.path.join(self.cf.dataset.path, self.cf.dataset.dataset_name+'.pkl.gz'))
            img_rows, img_cols = self.cf.dataset.img_shape

            # Reshape
            if K.image_dim_ordering() == 'th':
                x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
                x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            elif K.image_dim_ordering() == 'tf':
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

            # Normalize
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            # One hot encoding
            y_train2 = np.zeros((len(y_train), self.cf.dataset.n_classes), dtype='float32')
            for i, label in enumerate(y_train):
                y_train2[i, label] = 1.
            y_train = y_train2

            hist = self.model.fit(x=x_train,
                                  y=y_train,
                                  batch_size=self.cf.batch_size_train,
                                  validation_split=0.2,
                                  # samples_per_epoch=self.cf.dataset.n_images_train,
                                  nb_epoch=self.cf.n_epochs,
                                  verbose=1,
                                  callbacks=cb,
                                  # validation_data=valid_gen,
                                  # nb_val_samples=self.cf.dataset.n_images_valid,
                                  class_weight=None
                                  )
            print('   Training finished.')

            return hist
        else:
            return None

    ##
    # Predict the model
    def predict(self, test_gen, tag='pred'):
        if self.cf.pred_model:
            print('\n > Predicting the model...')
            # Load best trained model
            self.model.load_weights(self.cf.weights_file)

            # Create a data generator
            data_gen_queue, _stop, _generator_threads = GeneratorEnqueuer(self.test_gen, max_q_size=cf.max_q_size)
            ##data_gen_queue = GeneratorEnqueuer(test_gen, pickle_safe=True)

            # Process the dataset
            start_time = time.time()

            for _ in range(int(math.ceil(self.cf.dataset.n_images_train/float(self.cf.batch_size_test)))):

                ##added:
                ##data = None

                # Get data for this minibatch
                data = data_gen_queue.get()

                ##added:
                ##data = data_gen_queue.queue.get()

                x_true = data[0]
                y_true = data[1].astype('int32')

                # Get prediction for this minibatch
                y_pred = self.model.predict(x_true)

                # Compute the argmax
                y_pred = np.argmax(y_pred, axis=1)

                # Reshape y_true
                y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                             y_true.shape[3]))

                save_img3(x_true, y_true, y_pred, self.cf.savepath, 0,
                          self.cf.dataset.color_map, self.cf.dataset.classes,
                          tag+str(_), self.cf.dataset.void_class)

            # Stop data generator
            _stop.set()

            ##added
            ##data_gen_queue.stop()

            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Predicting time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))


    ##
    def SE_predict(self, test_gen, tag='SE_pred'):
        if self.cf.SE_pred_model:
            print('\n > Snapshot Ensembling, predicting using models from the ensemble0...')
            # Load models
            #self.model.load_weights(self.cf.weights_file)
            #model_list =  sorted(os.listdir(self.cf.savepath_SE_weights))
            #print(model_list)
            #print('AAAAAAAAAAAA')



            # ------------------------------------------
            # Create a data generator
            ## The task of an Enqueuer is to use parallelism to speed up preprocessing.
            ## This is done with processes or threads.

            ## same values from Save_results callback (?)
            nb_worker = 5
            max_q_size = 10


            # Load best trained model
            self.model.load_weights(self.cf.weights_file)

            enqueuer = GeneratorEnqueuer(test_gen, pickle_safe=True)
            enqueuer.start(nb_worker=nb_worker, max_q_size=max_q_size,
                           wait_time=0.05)

            # Process the dataset

            for _ in range(self.epoch_length):

                # Get data for this minibatch
                data = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        data = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.05)
                x_true = data[0]
                y_true = data[1].astype('int32')

                # Get prediction for this minibatch
                y_pred = self.model.predict(x_true)

                # Reshape y_true and compute the y_pred argmax
                if K.image_dim_ordering() == 'th':
                    y_pred = np.argmax(y_pred, axis=1)
                    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                                 y_true.shape[3]))
                else:
                    y_pred = np.argmax(y_pred, axis=3)
                    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                 y_true.shape[2]))
                # Save output images
                save_img3(x_true, y_true, y_pred, self.save_path, epoch,
                          self.color_map, self.classes, self.tag+str(_),
                          self.void_label, self.n_legend_rows)

            # Stop data generator
            if enqueuer is not None:
                enqueuer.stop()

            # ------------------------------------------















    # Test the model
    def test(self, test_gen):
        if self.cf.test_model:
            print('\n > Testing the model...')
            # Load best trained model
            self.model.load_weights(self.cf.weights_file)

            # Evaluate model
            start_time = time.time()
            ## Returns scalar test loss (if the model has no metrics) or list
            ## of scalars (if the model computes other metrics).
            test_metrics = self.model.evaluate_generator(test_gen,
                                                         self.cf.dataset.n_images_test,
                                                         max_q_size=self.cf.max_q_size,
                                                         nb_worker=self.cf.workers,
                                                         pickle_safe=True)
            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

            if self.cf.problem_type == 'detection':
                raise ValueError('Copy from master repository')

            elif self.cf.problem_type == 'segmentation':
                # Compute Jaccard per class

                ##test_metrics: see above
                ##model.metrics_names: ['loss', 'U9', 'U8', 'U5', 'U4', 'U7', 'U6', 'U1', 'U0', 'U3', 'U2', 'U10', 'I9', 'I8', 'I1', 'I0', 'I3', 'I2', 'I5', 'I4', 'I7', 'I6', 'I10', 'acc']

                metrics_dict = dict(zip(self.model.metrics_names, test_metrics))

                ## metrics_dict: {'U9': 8084.333333333333, 'U8': 54890.0, 'U5': 244498.66666666666, 'U4': 161890.33333333334, 'U7': 36734.333333333336, 'U6': 8936.0, 'U1': 531287.0, 'U0': 145011.33333333334, 'U3': 1708693.3333333333, 'U2': 5956.0, 'U10': 26039.0, 'I9': 0.0, 'I8': 0.0, 'I1': 0.0, 'I0': 0.0, 'I3': 485366.33333333331, 'I2': 0.0, 'I5': 0.0, 'I4': 0.0, 'I7': 0.0, 'I6': 0.0, 'I10': 0.0, 'acc': 0.28408331672350567, 'loss': 2.3173979123433432}


                I = np.zeros(self.cf.dataset.n_classes)
                U = np.zeros(self.cf.dataset.n_classes)
                jacc_percl = np.zeros(self.cf.dataset.n_classes)
                for i in range(self.cf.dataset.n_classes):
                    I[i] = metrics_dict['I'+str(i)]
                    U[i] = metrics_dict['U'+str(i)]
                    jacc_percl[i] = I[i] / U[i]
                    print ('   {:2d} ({:^15}): Jacc: {:6.2f}'.format(i,
                                                                     self.cf.dataset.classes[i],
                                                                     jacc_percl[i]*100))
                # Compute jaccard mean
                jacc_mean = np.nanmean(jacc_percl)
                print ('   Jaccard mean: {}'.format(jacc_mean))
