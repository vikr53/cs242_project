import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import datetime as dt
import numpy as np
import math
from mpi4py import MPI
import datetime
from quantizer import Quantize

#import torch
#import torchvision
#import torchvision.transforms as transforms
import sys
# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

def res_net_block(input_data, filters, conv_size):
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([x, input_data])
  x = layers.Activation('relu')(x)
  return x

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
N = nproc - 1 # one node is the server
rank = comm.Get_rank()

x_train_local = np.zeros((1,))
y_train_local = np.zeros((1,))   
x_test_local = np.zeros((1,))   
y_test_local = np.zeros((1,))   

optimizer = tf.keras.optimizers.SGD()
num_epoch = 100
alpha = 0.01 # learning rate

batch_size=8

# Loss metric
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

# Accuracy metric
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

if rank == 0:
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize input images
    x_train, x_test = tf.cast(x_train, tf.float32),  tf.cast(x_test, tf.float32)

    train_size = x_train.shape[0] # number of training samples
    test_size = x_test.shape[0] # number of testing samples

    # Partition training data
    split_train_idx = np.random.choice(train_size, (N, math.floor(train_size/N)), replace=False)
    
    # Similarly, partition test data
    # I. Split the validation dataset
    # split_test_idx = np.random.choice(test_size, (N, math.floor(test_size/N)), replace=False)

    d_xtrain = np.array((len(split_train_idx[0]),))
    d_ytrain = np.array((len(split_train_idx[0]),))
    
    # Communicate data partition (point-to-point)
    for n in range(1,N+1):
        x_train_local = np.array([x_train[idx] for idx in split_train_idx[n-1]])
        y_train_local = np.array([y_train[idx] for idx in split_train_idx[n-1]])
        
        # II. Don't split validation
        #x_test_local = np.array([x_test[idx] for idx in split_test_idx[n-1]])
        #y_test_local = np.array([y_test[idx] for idx in split_test_idx[n-1]])

        if n == 1:
            d_xtrain = x_train_local
            d_ytrain = y_train_local

        # comm.send([x_train_local, y_train_local, x_test_local, y_test_local], dest=n, tag=11)
        comm.send([x_train_local, y_train_local, x_test, y_test], dest=n, tag=11)
    
    # Instantiate model
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 8
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    # load weights
    model.load_weights("./chkpts/init_resnet8.ckpt")

    # dummy dataset to get number of steps
    dset = tf.data.Dataset.from_tensor_slices((d_xtrain, d_ytrain))
    dset = dset.shuffle(buffer_size=60000).batch(batch_size)
    
    # Aggregation
    for epoch in range(num_epoch):
        for step in enumerate(dset):            
            # obtain gradients from each node (assuming at least one other node)
            grad = comm.recv(source=1, tag=11)
            for n in range(2,N+1):
                data = comm.recv(source=n, tag=11)
                for j in range(len(data)):
                    grad[j] = np.array(grad[j]) + np.array(data[j])
            for j in range(len(grad)):
                grad[j] = (1.0/N) * grad[j]

            # Apply gradients
            #optimizer.learning_rate = alpha
            optimizer.apply_gradients(zip(grad, model.trainable_weights))

            ## NOT WORKING: Send new weights to nodes
            #weights = server_model.get_weights()
            #for n in range(1,N+1):
                #comm.send(weights, dest=n, tag=11)
            ## WORK AROUND: Send gradients to nodes
            for n in range(1, N+1):
                comm.send(grad, dest=n, tag=11)
else:
    # Receive partitioned data at node
    x_train_local, y_train_local, x_test_local, y_test_local = comm.recv(source=0, tag=11)
    x_test_local = tf.keras.applications.resnet.preprocess_input(x_test_local)    
    x_train_local = tf.keras.applications.resnet.preprocess_input(x_train_local)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_local, y_train_local))
    train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test_local, y_test_local))
    val_dataset = val_dataset.shuffle(buffer_size=20000).batch(batch_size)
    
    # Instantiate model
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 8
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    
    # model load weights
    model.load_weights("./chkpts/init_resnet8.ckpt")

    # Unfreeze Batch Norm layers                                                                              
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
    
    # Set up summary writers
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_time = "q_32"
    train_log_dir = 'logs/resnet8_tfprep8_lr0.01_daepoch_cor_alt'+str(rank)+'/' + current_time + '/train'
    test_log_dir = 'logs/resnet8_tfprep8_lr0.01_daepoch_cor_alt'+str(rank)+'/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    for epoch in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                loss = loss_fn(y_batch_train, y_pred)
                train_accuracy(y_batch_train, y_pred)
                train_loss(loss)
                #print("Step", step, loss)
                
                gradient_list = tape.gradient(loss, model.trainable_weights)
                q = Quantize()
                # Set the bitwidth here
                q.bitwidth = 32
                q_gradient_list = []
                for each_array in gradient_list:
                    q_w = q.quantize(each_array.numpy())
                    q_gradient_list.append(tf.convert_to_tensor(q_w))
                
                # TEST
                '''
                for each in range(len(q_gradient_list)):
                    print(q_gradient_list[each])
                    print("+++++++++++++++++++++++++++++")
                    print(gradient_list[each])
                    sys.exit()
                '''

                # Send QUANTIZED gradients to server
                comm.send(q_gradient_list, dest=0, tag=11)

                # Send gradients to server
                #comm.send(gradient_list, dest=0, tag=11)

            ## NOT WORKING: Receive and set weights from server
            #weights = comm.recv(source=0, tag=11)
            #model.set_weights(weights)
            
            ## WORK AROUND: Receive and apply gradients
            grad = comm.recv(source=0, tag=11)
            optimizer.learning_rate = alpha
            #optimizer.momentum = 0.9
            optimizer.apply_gradients(zip(grad, model.trainable_weights))

        print(f"Accuracy over epoch {train_accuracy.result()}")
        print(f"Loss over epoch {train_loss.result()}")

        # Log train metrics
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
        #train_acc = train_accuracy.result()
        #print(f"Accuracy over epoch {train_acc}")

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
                
        # Run validation
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training = False)
            val_loss(loss_fn(y_batch_val, val_logits))
            val_accuracy(y_batch_val, val_logits)

        # Log Validation metrics
        with test_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss.result(), step=epoch)
            tf.summary.scalar('val_accuracy', val_accuracy.result(), step=epoch)

        print("Validation acc: %.4f" % (float(val_accuracy.result()),))
    
        val_loss.reset_states()
        val_accuracy.reset_states()
