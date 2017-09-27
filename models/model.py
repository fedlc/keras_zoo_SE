# import os
import math
import time
import os
import numpy as np
from keras.engine.training import GeneratorEnqueuer
from tools.save_images import save_img3
from keras import backend as K
# from tools.yolo_utils import *

##
import h5py
import gc ## garbage collector
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
    def predict(self, test_gen, tag='pred', tag_gen=''):
        if self.cf.pred_model:
            print('\n > Predicting the single model using ' + tag_gen)
            checking_time_pred = time.time()

            test_gen.reset()

            #print("test_gen")
            #print(test_gen)

            ## get number of images in dataset
            nb_sample = len(test_gen.filenames[:])

            ## ----- CREATE hdf5 FILE TO SAVE y_pred AND y_true  -----

            ## create hdf5 file to save the predictions and the gt, to use them when testing
            y_file = h5py.File(self.cf.real_savepath + "/y_file_" + tag_gen +".hdf5", "w")

            ## provisional
            if self.cf.resize_valid != None:
                if (tag_gen == 'valid_gen'):
                    img_height, img_width = self.cf.resize_valid
                elif (tag_gen == 'test_gen'):
                    img_height, img_width = self.cf.resize_test
                else:
                    raise ValueError('Unknown data generator tag')
            else:
                img_height, img_width = (360,480)
                ## to change for cityscape (if we don't ask to resize)


            ## create datasets to save predictions and gt
            y_pred_dset = y_file.create_dataset("y_pred_dataset",
                                                     (nb_sample,img_height,img_width), dtype='i8')
            y_true_dset = y_file.create_dataset("y_true_dataset",
                                                     (nb_sample,img_height,img_width), dtype='i8')
            ## ACHTUNG: for SE, it will have to be dtype='f32' --NO!

            #print('\ny_pred_dset at the beginning')
            #print(type(y_pred_dset))
            #print(y_pred_dset.shape)
            #print(y_pred_dset.dtype)


            ## ----- LOAD MODEL WEIGHTS -----
            print('\nLoading model weights')
            # Load best trained model (or last? see camvid.py)
            weights_fl = os.path.join(self.cf.real_savepath, "weights.hdf5")
            self.model.load_weights(weights_fl)

            ## ----- PREDICT -----
            print('\nPredicting..')
            checking_time = time.time()
            y_pred = self.model.predict_generator(test_gen, val_samples=nb_sample,
                                max_q_size=10, nb_worker=1, pickle_safe=False)
                                #this combination works, not clear why
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            #print('\ny_pred right after prediction')
            #print(type(y_pred))
            #print(y_pred.shape)
            #print(y_pred.dtype)

            # Compute the y_pred argmax
            if K.image_dim_ordering() == 'th':
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.argmax(y_pred, axis=3)


            #print('\ny_pred after argmax')
            #print(type(y_pred))
            #print(y_pred.shape)
            #print(y_pred.dtype)

            ## ----- SAVE PREDICTIONS IN DATASET OF hdf5 FILE -----
            print('\nSaving predictions in hdf5 file')
            y_pred_dset[:] = y_pred[:]

            #print('\ny_pred_dset after assignment of y_pred')
            #print(type(y_pred_dset))
            #print(y_pred_dset.shape)
            #print(y_pred_dset.dtype)


            ## ----- LOAD x_true AND y_true IN BATCHES FROM enqueuer -----
            print('\nLoading x_true and y_true from enqueuer..')
            ## reset to restart from first batch
            test_gen.reset()

            ## same values from Save_results callback (?)
            nb_worker = 5
            max_q_size = 10

            ## batch size
            if (tag_gen == 'valid_gen'):
                batch_size = self.cf.batch_size_valid
            elif (tag_gen == 'test_gen'):
                batch_size = self.cf.batch_size_test
            else:
                raise ValueError('Unknown data generator tag')

            y_true = np.zeros((nb_sample,img_height,img_width), dtype='int8')
            x_true = np.zeros((nb_sample,img_height,img_width,3), dtype='float32')





            enqueuer = GeneratorEnqueuer(test_gen, pickle_safe=True)
            #print("\nenqueuer")
            #print(enqueuer)





            # Process the dataset
            nb_iterations = int(math.ceil(nb_sample/float(batch_size)))
            for _ in range(nb_iterations):
                checking_time = time.time()
                print("  Iteration " + str(_+1) + '/' + str(nb_iterations))

                ##here

                enqueuer.start(nb_worker=nb_worker, max_q_size=max_q_size,
                           wait_time=0.05)



                # Get data for this minibatch
                data = None

                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        #print('inside')
                        data = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.05)

                x_true_batch = data[0]
                #print('\nx_true_batch')
                #print(type(x_true_batch))
                #print(x_true_batch.shape)
                #print(x_true_batch[0])
                #print(np.max(x_true_batch))
                #print(np.min(x_true_batch))

                y_true_batch = data[1].astype('int32')
                #print('\ny_true_batch before squeeze')
                #print(type(y_true_batch))
                #print(y_true_batch.shape)

                y_true_batch = np.squeeze(y_true_batch, axis=3)
                #print('\ny_true_batch after squeeze')
                #print(type(y_true_batch))
                #print(y_true_batch.shape)

                # Reshape y_true_batch
                if K.image_dim_ordering() == 'th':
                    y_true_batch = np.reshape(y_true_batch, (y_true_batch.shape[0], y_true_batch.shape[2],
                                                 y_true_batch.shape[3]))
                else:
                    y_true_batch = np.reshape(y_true_batch, (y_true_batch.shape[0], y_true_batch.shape[1],
                                                 y_true_batch.shape[2]))

                #print('\ny_true_batch after reshape')
                #print(type(y_true_batch))
                #print(y_true_batch.shape)
                #print(y_true_batch.dtype)

                y_true[_*batch_size : (_*batch_size+len(y_true_batch))] = y_true_batch
                x_true[_*batch_size : (_*batch_size+len(x_true_batch))] = x_true_batch



                # Stop data generator
                if enqueuer is not None:
                    enqueuer.stop()
                del(y_true_batch)
                del(x_true_batch)
                #gc.collect()

                ## go to next batch
                if (_ < nb_iterations-1):
                    test_gen.next()

                print ('    time: ' + str( int(time.time()-checking_time)) + ' seconds')

            print('\nSaving y_true in hdf5 file')
            checking_time = time.time()
            y_true_dset[:] = y_true[:]
            ## close hdf5 file (and save it)
            y_file.close()
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')
            #print('\ny_true_dset after assignment of y_true')
            #print(type(y_true_dset))
            #print(y_true_dset.shape)
            #print(y_true_dset.dtype)



            #enq stop here

            if (tag_gen == 'valid_gen'):
                img_savepath = self.cf.savepath_pred_valid_gen
            elif (tag_gen == 'test_gen'):
                img_savepath = self.cf.savepath_pred_test_gen
            else:
                raise ValueError('Unknown data generator tag')


            ## ----- SAVE PREDICTIONS AS IMAGES -----
            print('\nSaving ' + str(self.cf.nb_pred_images_to_save) + ' prediction images..')
            checking_time = time.time()
            # Save output images
            nb_save = self.cf.nb_pred_images_to_save
            save_img3(image_batch=x_true[0:nb_save], mask_batch=y_true[0:nb_save], output=y_pred[0:nb_save],
                      out_images_folder=img_savepath, epoch=-1,
                      color_map=self.cf.dataset.color_map, classes=self.cf.dataset.classes,
                      tag=tag, void_label=self.cf.dataset.void_class, n_legend_rows=1,
                      tag2='prediction_images')
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

        print('\nSingle model prediction of ' + tag_gen + ' terminated')
        print ('  time: ' + str( int(time.time()-checking_time_pred)) + ' seconds')



    ##
    def SE_predict(self, test_gen, tag='SE_pred', tag_gen=''):
        if self.cf.SE_pred_model:
            print('\n > Snapshot Ensembling predictions')
            checking_time_pred = time.time()

            test_gen.reset()


            ## provisional
            if self.cf.resize_valid != None:
                if (tag_gen == 'valid_gen'):
                    img_height, img_width = self.cf.resize_valid
                elif (tag_gen == 'test_gen'):
                    img_height, img_width = self.cf.resize_test
                else:
                    raise ValueError('Unknown data generator tag')
            else:
                img_height, img_width = (360,480)
                ## to change for cityscape


            model_list =  sorted(os.listdir(self.cf.savepath_SE_weights))
            nb_models = len(model_list)
            #print(model_list)
            #print(nb_models)

            ## get number of images in dataset
            nb_sample = len(test_gen.filenames[:]) #101

            ## ----- CREATE hdf5 FILE TO SAVE y_pred AND y_true  -----

            ## create hdf5 file to save the predictions and the gt, to use them when testing
            y_file = h5py.File(self.cf.real_savepath + "/y_file_SE_" + tag_gen +".hdf5", "w")

            ## create datasets to save predictions and gt
            y_pred_dset = y_file.create_dataset("y_pred_dataset",
                                                     (nb_sample,img_height,img_width), dtype='i8')
            y_true_dset = y_file.create_dataset("y_true_dataset",
                                                     (nb_sample,img_height,img_width), dtype='i8')


            #print('\ny_pred_dset at the beginning')
            #print(type(y_pred_dset))
            #print(y_pred_dset.shape)
            #print(y_pred_dset.dtype)


            ## ----- CREATE hdf5 FILE TO SAVE THE SOFTMAX PREDICTIONS  -----

            ## create hdf5 file to save the predictions and the gt, to use them when testing
            #y_pred_file = h5py.File(self.cf.real_savepath + "/y_pred_file.hdf5", "w")

            ## create datasets to save predictions and gt
            #y_pred_dset = y_pred_file.create_dataset("y_pred_dataset",
            #                                         (nb_sample,360,480,11), dtype='float32')
            ## ACHTUNG: for SE, it will have to be dtype='f32'


            classes_dict = self.cf.dataset.classes
            #print('a')
            #print(len(classes_dict))

            if 'void' in classes_dict.values():
                keys_dict = {v: k for k, v in classes_dict.iteritems()}
                key_void = keys_dict['void']

                classes_dict = {x: classes_dict[x] for x in classes_dict if classes_dict[x] != 'void'}

            nb_classes = len(classes_dict)

            #print(nb_classes)

            #prova con concatenate
            y_pred = np.zeros((nb_models,nb_sample,img_height,img_width,nb_classes), dtype='float32')

            print('\ny_pred')
            print(type(y_pred))
            print(y_pred.shape)
            print(y_pred.dtype)


            ## ----- LOAD MODEL WEIGHTS AND PREDICT-----
            print('Loading model weights, and predicting')

            i = 0
            for model_file in model_list:
                test_gen.reset()
                # Load best trained model (or last? see camvid.py)
                print('  Loading model ' + model_file)
                weights_fl = os.path.join(self.cf.savepath_SE_weights, model_file)
                self.model.load_weights(weights_fl)

                print('  Predicting with ' + model_file)
                checking_time = time.time()
                y_pred[i] = self.model.predict_generator(test_gen, val_samples=nb_sample,
                            max_q_size=10, nb_worker=1, pickle_safe=False)
                            #this combination works, not clear why
                i += 1
                print ('    time: ' + str( int(time.time()-checking_time)) + ' seconds')


            #print('\ny_pred right after prediction')
            #print(type(y_pred))
            #print(y_pred.shape)
            #print(y_pred.dtype)
            #print(y_pred)
            print('Averaging models predictions')
            checking_time_aver = time.time()
            # model weights
            w = np.array(self.cf.SE_model_weights)

            eps = 0.00001
            if (len(w)!=nb_models or not (np.sum(w)<1+eps and np.sum(w)>1-eps)):
                raise Exception('Model weights are incorrect')

            for i in range(len(y_pred)):
                checking_time = time.time()
                y_pred[i] = y_pred[i]*w[i]
                print ('  time moltiplication with w: ' + str(i) + ' ' + str( int(time.time()-checking_time)) + ' seconds')

            checking_time = time.time()
            y_pred = np.sum(y_pred, axis=0)
            print ('  time sum: ' + str( int(time.time()-checking_time)) + ' seconds')
            print ('  total time averaging: ' + str( int(time.time()-checking_time_aver)) + ' seconds')

            #y_pred = np.average(y_pred, axis=0, weights=w)
            #print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            #print('\ny_pred after average')
            #print(type(y_pred))
            #print(y_pred.shape)
            #print(y_pred.dtype)


            print('Computing argmax')
            checking_time = time.time()
            # Compute the y_pred argmax
            if K.image_dim_ordering() == 'th':
                y_pred = np.argmax(y_pred, axis=1)
            else:
                y_pred = np.argmax(y_pred, axis=3)
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')


            #print('\ny_pred after argmax')
            #print(type(y_pred))
            #print(y_pred.shape)
            #print(y_pred.dtype)

            ## ----- SAVE PREDICTIONS IN DATASET OF hdf5 FILE -----
            print('Saving predictions')
            checking_time = time.time()
            y_pred_dset[:] = y_pred[:]
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            #print('\ny_pred_dset after assignment of y_pred')
            #print(type(y_pred_dset))
            #print(y_pred_dset.shape)
            #print(y_pred_dset.dtype)


            ## ----- LOAD x_true AND y_true IN BATCHES FROM enqueuer -----
            print('\nLoading x_true and y_true from enqueuer..')
            print('  Preparation')
            checking_time = time.time()

            ## reset to restart from first batch
            test_gen.reset()

            ## same values from Save_results callback (?)
            nb_worker = 5
            max_q_size = 10

            ## batch size
            if (tag_gen == 'valid_gen'):
                batch_size = self.cf.batch_size_valid
            elif (tag_gen == 'test_gen'):
                batch_size = self.cf.batch_size_test
            else:
                raise ValueError('Unknown data generator tag')

            y_true = np.zeros((nb_sample,img_height,img_width), dtype='int8')
            x_true = np.zeros((nb_sample,img_height,img_width,3), dtype='float32')








            enqueuer = GeneratorEnqueuer(test_gen, pickle_safe=True)
            #print("\nenqueuer")
            #print(enqueuer)

            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            # Process the dataset
            nb_iterations = int(math.ceil(nb_sample/float(batch_size)))
            for _ in range(nb_iterations):
                checking_time = time.time()
                print("  Iteration " + str(_+1) + '/' + str(nb_iterations))

                enqueuer.start(nb_worker=nb_worker, max_q_size=max_q_size,
                               wait_time=0.05)

                # Get data for this minibatch
                data = None

                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        #print('inside')
                        data = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.05)

                x_true_batch = data[0]
                #print('\nx_true_batch')
                #print(type(x_true_batch))
                #print(x_true_batch.shape)

                y_true_batch = data[1].astype('int32')
                #print('\ny_true_batch before squeeze')
                #print(type(y_true_batch))
                #print(y_true_batch.shape)

                y_true_batch = np.squeeze(y_true_batch, axis=3)
                #print('\ny_true_batch after squeeze')
                #print(type(y_true_batch))
                #print(y_true_batch.shape)

                # Reshape y_true_batch
                if K.image_dim_ordering() == 'th':
                    y_true_batch = np.reshape(y_true_batch, (y_true_batch.shape[0], y_true_batch.shape[2],
                                                 y_true_batch.shape[3]))
                else:
                    y_true_batch = np.reshape(y_true_batch, (y_true_batch.shape[0], y_true_batch.shape[1],
                                                 y_true_batch.shape[2]))

                #print('\ny_true_batch after reshape')
                #print(type(y_true_batch))
                #print(y_true_batch.shape)
                #print(y_true_batch.dtype)

                y_true[_*batch_size : (_*batch_size+len(y_true_batch))] = y_true_batch
                x_true[_*batch_size : (_*batch_size+len(x_true_batch))] = x_true_batch

                # Stop data generator
                if enqueuer is not None:
                    enqueuer.stop()
                del(y_true_batch)
                del(x_true_batch)

                ## go to next batch
                if (_ < nb_iterations-1):
                    test_gen.next()

                print ('    time: ' + str( int(time.time()-checking_time)) + ' seconds')


            print('\nSaving y_true in hdf5 file')
            checking_time = time.time()
            y_true_dset[:] = y_true[:]
            ## close h5py file (and save it)
            y_file.close()
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            #print('\ny_true_dset after assignment of y_true')
            #print(type(y_true_dset))
            #print(y_true_dset.shape)
            #print(y_true_dset.dtype)






            if (tag_gen == 'valid_gen'):
                img_savepath = self.cf.savepath_pred_SE_valid_gen
            elif (tag_gen == 'test_gen'):
                img_savepath = self.cf.savepath_pred_SE_test_gen
            else:
                raise ValueError('Unknown data generator tag')



            ## ----- SAVE PREDICTIONS AS IMAGES -----
            print('\nSaving ' + str(self.cf.nb_pred_images_to_save) + ' prediction images..')
            checking_time = time.time()

            # Save output images
            nb_save = self.cf.nb_pred_images_to_save
            save_img3(image_batch=x_true[0:nb_save], mask_batch=y_true[0:nb_save], output=y_pred[0:nb_save],                      out_images_folder=img_savepath, epoch=-1,
                      color_map=self.cf.dataset.color_map, classes=self.cf.dataset.classes,
                      tag=tag, void_label=self.cf.dataset.void_class, n_legend_rows=1,
                      tag2='prediction_images')
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

        print('\nSE prediction of ' + tag_gen + ' terminated')
        print ('  time: ' + str( int(time.time()-checking_time_pred)) + ' seconds')



    # Test the model
    def test(self, test_gen, tag='', tag_gen=''):
        if (self.cf.test_model or self.cf.SE_test_model):
            print('\n > ' + tag + '-testing the model with predictions from ' + tag_gen)
            checking_time_test = time.time()

            test_gen.reset()

            ## get number of images in dataset TO DELETE?
            nb_sample = len(test_gen.filenames[:])

            ## ----- Load y_pred and y_true from the hdf5 file -----

            print('\nLoading data...')
            checking_time = time.time()
            if tag=='test_SE':
                y_file = h5py.File(self.cf.real_savepath + "/y_file_SE_" + tag_gen +".hdf5", 'r')
            else:
                y_file = h5py.File(self.cf.real_savepath + "/y_file_" + tag_gen +".hdf5", 'r')


            y_pred_dset = y_file['.']['y_pred_dataset'].value
            y_true_dset = y_file['.']['y_true_dataset'].value

            #print('\ny_pred_dset')
            #print(type(y_pred_dset))
            #print(y_pred_dset.shape)

            #print("\ny_pred_dset[0]")
            #print(type(y_pred_dset))
            #print(y_pred_dset[0].shape)

            y_file.close()

            #y_pred = np.zeros((nb_sample,360,480), dtype=np.int8)
            #y_true = np.zeros((nb_sample,360,480), dtype=np.int8)


            y_pred = np.array(y_pred_dset)
            y_true = np.array(y_true_dset)

            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            #print('\ny_pred')
            #print(type(y_pred))
            #print(y_pred.shape)
            #print(y_pred.dtype)

            #print("\ny_true")
            #print(type(y_true))
            #print(y_true.shape)
            #print(y_true.dtype)

            ## ----- COMPUTE ACCURACY -----
            print('\nLoading classes')
            checking_time = time.time()
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            #acc = accuracy_score(y_true.flatten(), y_pred.flatten(), normalize=True)
            #jac = jaccard_similarity_score(y_true.flatten(), y_pred.flatten(), normalize=True)


            classes_dict = self.cf.dataset.classes

            print('\nComputing confusion matrix')

            confusion_matr = confusion_matrix(y_true.flatten(), y_pred.flatten())
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            print(confusion_matr)

            print('\nremoving void class')
            checking_time = time.time()


            if 'void' in classes_dict.values():
                keys_dict = {v: k for k, v in classes_dict.iteritems()}
                key_void = keys_dict['void']
                print(classes_dict)

                confusion_matr = np.delete(confusion_matr, key_void, 0)
                confusion_matr = np.delete(confusion_matr, key_void, 1)

                classes_dict = {x: classes_dict[x] for x in classes_dict if classes_dict[x] != 'void'}

            #report = classification_report(y_true.flatten(), y_pred.flatten(), [0,1,2,3,4,5,6,7,8,9,10], classes_dict.values())
            nb_classes = len(classes_dict)
            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')

            print('\nComputing accuracy, Jaccard index, recall and precision')
            checking_time = time.time()

            ## Compute true positive, true negative, etc

            TP = np.zeros((nb_classes), dtype='int')
            TN = np.zeros((nb_classes), dtype='int')
            FP = np.zeros((nb_classes), dtype='int')
            FN = np.zeros((nb_classes), dtype='int')

            sum_rows = np.sum(confusion_matr, axis=1)
            sum_columns = np.sum(confusion_matr, axis=0)

            #print("classes_dict")
            #print(classes_dict)

            #print('\nsum_rows')
            #print(sum_rows)
            #print('\nsum_columns')
            #print(sum_columns)

            for i in range(nb_classes):
                TP[i] = confusion_matr[i][i]
                TN[i] = np.sum(confusion_matr) - sum_rows[i] - sum_columns[i] + TP[i]
                FP[i] = sum_columns[i] - TP[i]
                FN[i] = sum_rows[i] - TP[i]

            # Compute metrics
            acc_percl = np.zeros((nb_classes), dtype='float32')
            jacc_percl = np.zeros((nb_classes), dtype='float32')
            recall_percl =  np.zeros((nb_classes), dtype='float32')
            precision_percl =  np.zeros((nb_classes), dtype='float32')

            acc_percl = (TP+TN)/(TP+TN+FP+FN).astype(float)
            jacc_percl = TP/(TP+FN+FP).astype(float)
            recall_percl = TP/(TP+FN).astype(float)
            precision_percl = TP/(TP+FP).astype(float)

            # Set to 0 possible NaN elements
            acc_percl[np.isnan(acc_percl)] = 0.
            jacc_percl[np.isnan(jacc_percl)] = 0.
            recall_percl[np.isnan(recall_percl)] = 0.
            precision_percl[np.isnan(precision_percl)] = 0.

            # Compute metrics means
            acc_mean = np.nanmean(acc_percl)
            jacc_mean = np.nanmean(jacc_percl)
            recall_mean = np.nanmean(recall_percl)
            precision_mean = np.nanmean(precision_percl)

            print ('  time: ' + str( int(time.time()-checking_time)) + ' seconds')


            if tag=='test_SE':
                file_results = '/SE_Model_results_' + tag_gen + '.txt'
                w = str(self.cf.SE_model_weights)
            else:
                file_results = '/single_Model_results_' + tag_gen + '.txt'
                w = 'We are testing a single model'


            # Print the results
            print('\t\t\tJaccard\t  accur\t  precis\t  recall')
            for i in range(nb_classes):
                print('   {:2d} ({:^15}): {:6.2f}    {:6.2f}    {:6.2f}    {:6.2f}'.format(i,
                                                                 classes_dict[i],
                                                                 jacc_percl[i]*100,
                                                                 acc_percl[i]*100,
                                                                 precision_percl[i]*100,
                                                                 recall_percl[i]*100))

            print('\n   Jaccard mean:  {}'.format(jacc_mean))
            print('   Accuracy mean: {}\n'.format(acc_mean))
            print('   Precision mean: {}\n'.format(precision_mean))
            print('   Recall mean: {}\n'.format(recall_mean))

            #print('\n\nClassification Report\n')
            #print(report)
            print('\n\nModels weights: {}\n'.format(w))


            # Save the results
            with open(self.cf.real_savepath + file_results, 'w') as f:
                f.write('\t\t\tJaccard\t  accur\t  precis\t  recall\n')
                for i in range(nb_classes):
                    f.write('   {:2d} ({:^15}): {:6.2f}    {:6.2f}    {:6.2f}    {:6.2f}\n'.format(i,
                                                                     classes_dict[i],
                                                                     jacc_percl[i]*100,
                                                                     acc_percl[i]*100,
                                                                     precision_percl[i]*100,
                                                                     recall_percl[i]*100))
                f.write('\n   Jaccard mean:  {}\n'.format(jacc_mean))
                f.write('   Accuracy mean: {}\n'.format(acc_mean))
                f.write('   Precision mean: {}\n'.format(precision_mean))
                f.write('   Recall mean: {}\n'.format(recall_mean))

                #f.write('\n\n\nClassification Report\n')
                #f.write(report)
                f.write('\n\nModels weights: {}\n'.format(w))

        print('\n' + tag + '-test with predictions from ' + tag_gen + ' terminated')
        print ('  time: ' + str( int(time.time()-checking_time_test)) + ' seconds')



    """
            OLD VERSION
            ## ----- LOAD y_true FROM enqueuer -----
            ## same values from Save_results callback (?)
            nb_worker = 5
            max_q_size = 10

            enqueuer = GeneratorEnqueuer(test_gen, pickle_safe=True)
            print("\nenqueuer")
            print(enqueuer)
            enqueuer.start(nb_worker=nb_worker, max_q_size=max_q_size,
                           wait_time=0.05)

            # Get data for this minibatch
            data = None

            print("enqueuer.is_running()")
            print(enqueuer.is_running())

            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    print('inside: enqueuer running and not empty')
                    data = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.05)

            y_true = data[1].astype('int32')
            print('\ny_true')
            print(type(y_true))
            print(y_true.shape)
            print(y_true)

            y_true_resized = np.squeeze(y_true)
            print('\ny_true_resized')
            print(type(y_true_resized))
            print(y_true_resized.shape)
            print(y_true_resized)


    """
    """
        OLD VERSION OF test
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

    """




    """
            OLD VERSION 2

            ## WE NEED y_true
            import skimage.io as io
            # change 100
            y_true = np.array(io.ImageCollection(self.cf.dataset.path_valid_mask + '/*'))

            print('\ny_true')
            print(type(y_true))
            print(y_true.shape)
            print(y_true.dtype)

            # change 100
            #x_true = np.array(io.ImageCollection(self.cf.dataset.path_valid_img + '/*'))[0:101]
            x_true = np.array(io.ImageCollection(self.cf.dataset.path_valid_img + '/*'))

            print('x_true')
            print(len(x_true))
            print(type(x_true))
            print(x_true.shape)
            print(x_true.dtype)


    """


    """
            Old version :(

            print("test_gen")
            print(test_gen)

            ## ----- CREATE hdf5 FILE TO SAVE y_pred  -----

            ## get number of images in dataset
            nb_sample = len(test_gen.filenames[:])

            ## create hdf5 file to save the predictions, to use them when testing
            y_pred_file = h5py.File(self.cf.real_savepath + "/y_pred.hdf5", "w")
            y_pred_dset = y_pred_file.create_dataset("y_pred_dataset", (nb_sample,360,480), dtype='i8')
            ## ACHTUNG: for SE, it will have to be dtype='f32'

            ## ----- LOAD MODEL WEIGHTS -----

            # Load best trained model (or last? see camvid.py)
            weights_fl = os.path.join(self.cf.real_savepath, "weights.hdf5")
            self.model.load_weights(weights_fl)


            #y_pred_prova = self.model.predict_generator(test_gen, val_samples=100,
            #                    max_q_size=10,
            #                  nb_worker=5, pickle_safe=True)
            print('\ny_pred_prova')
            print(type(y_pred_prova))
            print(y_pred_prova.shape)
            print(y_pred_prova.dtype)
            #print(y_pred_prova)

            #quit()



            ## ----- LOAD x_true AND y_true FROM enqueuer -----

            ## same values from Save_results callback (?)
            nb_worker = 5
            max_q_size = 10

            ## batch size
            if (tag_gen == 'valid_gen'):
                batch_size = self.cf.batch_size_valid
            elif (tag_gen == 'test_gen'):
                batch_size = self.cf.batch_size_test
            else:
                raise ValueError('Unknown data generator tag')

            # Process the dataset
            nb_iterations = int(math.ceil(nb_sample/float(batch_size)))
            for _ in range(nb_iterations):

                print("\n\nIteration " + str(_+1) + '/' + str(nb_iterations))

                enqueuer = GeneratorEnqueuer(test_gen, pickle_safe=True)
                print("\nenqueuer")
                print(enqueuer)
                enqueuer.start(nb_worker=nb_worker, max_q_size=max_q_size,
                               wait_time=0.05)

                # Get data for this minibatch
                data = None

                print("enqueuer.is_running()")
                print(enqueuer.is_running())

                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        print('inside')
                        data = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.05)

                x_true = data[0]
                print('\nx_true')
                print(type(x_true))
                print(x_true.shape)
                #print(x_true)

                y_true = data[1].astype('int32')
                print('\ny_true')
                print(type(y_true))
                print(y_true.shape)

                ## ----- MAKE PREDICTIONS -----

                # Get prediction for this minibatch
                ## this predict is from keras since model is a Keras Model
                y_pred = self.model.predict(x_true)
                print('\ny_pred')
                print(type(y_pred))
                print(y_pred.shape)
                print(y_pred.dtype)
                print(y_pred)



                ## both x_true and y_true contain batch_size_valid images ("valid" if test_gen==valid_gen)
                ## x_true is a numpy.ndarray with shape (10, 360, 480, 3)
                ## y_true is a numpy.ndarray with shape (10, 360, 480, 1)
                ## y_pred is a numpy.ndarray with shape (10, 360, 480, 11)





                fff = open( 'y_pred_ep' + str(epoch) + '.py', 'w' )

                for aaa in range(y_pred.shape[0]):
                    for bbb in range(y_pred.shape[1]):
                        for ccc in range(y_pred.shape[2]):
                            for ddd in range(y_pred.shape[3]):
                                eee = y_pred[aaa][bbb][ccc][ddd]
                                if (eee > 0 and eee < 1):
                                    print>>fff, eee

                fff.close()

                # Reshape y_true and compute the y_pred argmax
                if K.image_dim_ordering() == 'th':
                    y_pred = np.argmax(y_pred, axis=1)
                    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[2],
                                                 y_true.shape[3]))
                else:
                    y_pred = np.argmax(y_pred, axis=3)
                    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                 y_true.shape[2]))

                print('\ny_pred after argmax')
                print(type(y_pred))
                print(y_pred.shape)
                print(y_pred)

                ## ----- SAVE PREDICTIONS IN DATASET OF hdf5 FILE -----

                ##y_pred_dset[0:9] = y_pred

                y_pred_dset[_*batch_size : (_*batch_size+len(y_pred))] = y_pred
                ##
                print('\ny_pred_dset')
                print(type(y_pred_dset))
                print(y_pred_dset.shape)

                ## ----- SAVE PREDICTIONS AS IMAGES -----

                # Save output images
                ##
                save_img3(image_batch=x_true, mask_batch=y_true, output=y_pred,
                          out_images_folder=self.cf.savepath_pred, epoch=-1,
                          color_map=self.cf.dataset.color_map, classes=self.cf.dataset.classes,
                          tag=tag+str(_), void_label=self.cf.dataset.void_class, n_legend_rows=1)

                if (_ < nb_iterations-2):
                    test_gen.next()

            ## ----- CLOSE RUNNING STUFF -----
            # Stop data generator
            if enqueuer is not None:
                enqueuer.stop()

            ## close h5py file (and save it)
            y_pred_file.close()


    """

    """
    OLD VERSION OF predict
    # Predict the model
    def predict(self, test_gen, tag='pred'):
        if self.cf.pred_model:
            print('\n > Predicting the model...')
            # Load best trained model
            self.model.load_weights(self.cf.weights_file)

            # Create a data generator
            data_gen_queue, _stop, _generator_threads = GeneratorEnqueuer(self.test_gen, max_q_size=cf.max_q_size)

            # Process the dataset
            start_time = time.time()
            for _ in range(int(math.ceil(self.cf.dataset.n_images_train/float(self.cf.batch_size_test)))):

                # Get data for this minibatch
                data = data_gen_queue.get()
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

            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Predicting time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))


    """
