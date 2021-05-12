import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import datetime as dt
import numpy as np
import math
from mpi4py import MPI
import datetime
from quantization.quantizer import Quantize
from resnet_generator import ResNetForCIFAR10

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

### EXPERIMENT CONFIG ######################################
num_epoch = 70
alpha = 0.01 # learning rate

batch_size=8

base_model = "resnet" # available models: "FLNet, LeNet"

fbk = False # whether to use feedback error correction or not

## CHOOSE COMPRESSION SCHEME
# I. TOPK
topk = True
k = 190
k_decay = "None" # None if no decay

# II. QUANT
quant = 32 # quantization bit-width: if 32, then no quant
#############################################################

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
            
            ## NOT WORKING: Send new weights to nodes
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
    if base_model == "flnet":
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
    elif base_model == "lenet":
        inputs = keras.Input(shape=(32, 32, 3),name="input")
        x = layers.Conv2D(6, 3, activation='relu',name="layer1")(inputs)
        x = layers.MaxPooling2D(3)(x)
        x = layers.Conv2D(16, 3, activation='relu',name="layer2")(x)
        x = layers.MaxPooling2D(3)(x)
        x = layers.Flatten()(x)
    
        x = layers.Dense(120, activation='relu',name="FC1")(x)
        x = layers.Dense(84, activation='relu',name="FC2")(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        # load weights
        model.load_weights("./chkpts/init_lenet.ckpt")
    elif base_model == "resnet":
        weight_decay = 1e-4
        num_blocks = 9
        name = "resnet"
        model = ResNetForCIFAR10(input_shape=(32, 32, 3),name=name,block_layers_num=num_blocks, classes=10, weight_decay=weight_decay)
        
        #load weights
        model.load_weights("./chkpts/init_resnet.ckpt")
    else:
        print("Incorrect NN choice. Please choose either FLNet, lenet, or resnet!!")
        sys.exit()

    # Unfreeze Batch Norm layers                                                                              
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
    
    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = "logs/"+base_model+"/"
    if fbk:
        base += "fbk/"
    if topk:
        base += "topk/"
    if quant < 32:
        base += "quant_"+str(quant)+"/"
    if k_decay != None:
        base += 'kdecay_'+k_decay+str(k)+'/'
    else:
        base += 'k_'+str(k)+'/'

    train_log_dir = base+str(rank)+'/' + current_time + '/train'
    #grad_var_log_dir = base+'k_'+str(k)+'/'+str(rank)+'/' + current_time + '/grad_var'
    test_log_dir = base+str(rank)+'/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    #grad_var_writer = tf.summary.create_file_writer(grad_var_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    if fbk:
        r = []
        u = []

    for epoch in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch}")
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                loss = loss_fn(y_batch_train, y_pred)
                train_accuracy(y_batch_train, y_pred)
                train_loss(loss)
                
                grad = tape.gradient(loss, model.trainable_weights)

                concat_grads = tf.zeros((0,))
                grad_elem_len = []
                num_elem_in_grad = len(grad)
                grad_elem_shapes = []

                if fbk:
                    # Get grad elem shapes
                    for i in range(num_elem_in_grad):
                        grad_elem_shapes.append(grad[i].shape)
                        if step == 0 and epoch == 0:
                            r.append(np.zeros(grad_elem_shapes[i]))
                            u.append(np.zeros(grad_elem_shapes[i]))

                    for i in range(num_elem_in_grad):
                        u[i] = alpha*grad[i] + r[i]
                  
                    for i in range(num_elem_in_grad):
                        flattened = tf.reshape(u[i], -1) #flatten
                        concat_grads = tf.concat((concat_grads, flattened), 0)                        
                else:
                    for j in range(num_elem_in_grad):
                        flattened = tf.reshape(grad[j], -1) #flatten
                        grad_elem_len.append(len(flattened))
                        grad_elem_shapes.append(grad[j].shape)
                        concat_grads = tf.concat((concat_grads, flattened), 0)
                
                # Compute top-k of grad/u
                grad_tx = []
                if topk:
                    k_epoch = k
                    if k_decay == "lin":
                        n_epoch = 20
                        kf = 1
                        k_epoch = int(k - math.floor((epoch)*(k-kf)/(num_epoch)))

                    top_val, top_idx = tf.math.top_k(tf.math.abs(concat_grads), k_epoch)
                    k_val, k_idx = top_val[-1], top_idx[-1]
                    top_k_grad = [np.zeros(grad_elem_shapes[i]) for i in range(len(grad_elem_shapes))]
                    
                    
                    for i in range(num_elem_in_grad):
                        threshold = tf.fill(grad_elem_shapes[i], k_val)
                        if fbk:
                          mask = tf.math.abs(u[i]) < threshold
                        else:
                          mask = tf.math.abs(grad[i]) < threshold

                        elems_equal = tf.equal(mask, False)
                        as_int = tf.cast(elems_equal, tf.int32)
                        count = tf.reduce_sum(as_int)          

                        # Feedback error correction
                        if fbk:
                            np_u = np.array(u[i])
                            top_k_grad[i] = np.where(mask, 0.0 , np_u)
                            r[i] = u[i] - top_k_grad[i]
                        else:
                            np_grad = np.array(grad[i])
                            top_k_grad[i] = np.where(mask, 0.0 , np_grad)

                    grad_tx = top_k_grad

                # Compute quant of topk (switch order if needed)
                if quant < 32:
                    if fbk:
                        np_u = np.array(u)
                        
                        q = Quantize()
                        q.bitwidth = quant
                        q_np_u = []

                        for each_idx in range(len(np_u)):
                            q_w = q.quantize(np_u[each_idx])
                            q_np_u.append(q_w)

                            # Feedback error correction
                            r[each_idx] = u[each_idx] - q_np_u[each_idx]
                        grad_tx = q_np_u
                    else:
                        np_grad = np.array(grad)
                
                        q = Quantize()
                        q.bitwidth = quant
                        q_np_grad = []

                        for each_idx in range(len(np_grad)):
                            q_w = q.quantize(np_grad[each_idx])
                            q_np_grad.append(q_w)
                        grad_tx = q_np_grad
                        
                # Send gradients to server
                comm.send(grad_tx, dest=0, tag=11)

            ## NOT WORKING: Receive and set weights from server
            #weights = comm.recv(source=0, tag=11)
            #model.set_weights(weights)
            
            ## WORK AROUND: Receive and apply gradients
            grad_rx = comm.recv(source=0, tag=11)

            if fbk:
                optimizer.learning_rate = 1 # already applied learning rate
            else:
                optimizer.learning_rate = alpha
            #optimizer.momentum = 0.9
            optimizer.apply_gradients(zip(grad_rx, model.trainable_weights))

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
