import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import datetime as dt
import numpy as np
import math
from mpi4py import MPI
import datetime
from quantization.quantizer import Quantize
from models import *
from sampler import *

import sys
# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
  
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

batch_size=1

base_model = "flnet" # available models: "FLNet, ResNet9"

fbk = False # whether to use feedback error correction or not

sample = "non_iid"
## CHOOSE COMPRESSION SCHEME
# I. TOPK
topk = False
k = 668426
kf = 1
k_decay = None # None if no decay

# II. QUANT
quant = 32 # quantization bit-width: if 32, then no quant
#############################################################

# Loss metric
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

# Accuracy metric
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

# Grad var
grad_var_l2 = tf.keras.metrics.Mean('grad_var_l2', dtype=tf.float32)
grad_var_l1 = tf.keras.metrics.Mean('grad_var_l1', dtype=tf.float32)

if rank == 0:
    if sample == "non_iid":
        sampler = Sampler(False, N, "cifar10")
    else:
        sampler = Sampler(True, N, "cifar10")
        sampler.sample_iid(comm)
    
    # Aggregation
    for epoch in range(num_epoch):
        if sample == "non_iid":
            sampler.sample_noniid(comm)

        # dummy dataset to get number of steps
        dset = tf.data.Dataset.from_tensor_slices((sampler.x_train_local, sampler.y_train_local))
        dset = dset.shuffle(buffer_size=60000).batch(batch_size)
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
    if sample == "iid":
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
      model = flnet_init()
    elif base_model == "resnet9":
      model = resnet_init()
    else:
      print("Incorrect NN choice. Please choose either FLNet, lenet, or resnet!!")
      sys.exit()

    # Unfreeze Batch Norm layers                                                                              
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
    
    # Set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = "logs/"+base_model+"/" + sample + "/"
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
    
    if rank == 3:
        train_log_dir = base+str(rank)+'/schedulelr' + current_time + '/train'
        grad_var_l2_log_dir = base+str(rank)+'/schedulelr' + current_time + '/grad_var_l2'
        grad_var_l1_log_dir = base+str(rank)+'/schedulelr' + current_time + '/grad_var_l1'
        test_log_dir = base+str(rank)+'/schedulelr' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        grad_var_l2_writer = tf.summary.create_file_writer(grad_var_l2_log_dir)
        grad_var_l1_writer = tf.summary.create_file_writer(grad_var_l1_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    if fbk:
        r = []
        u = []
    prev_grad = []
    for epoch in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch}")
        if sample == "non_iid":
            # Receive dataset for this epoch
            x_train_local, y_train_local, x_test_local, y_test_local = comm.recv(source=0, tag=11)
            x_test_local = tf.keras.applications.resnet.preprocess_input(x_test_local)    
            x_train_local = tf.keras.applications.resnet.preprocess_input(x_train_local)

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train_local, y_train_local))
            train_dataset = train_dataset.batch(batch_size) # don't shuffle

            val_dataset = tf.data.Dataset.from_tensor_slices((x_test_local, y_test_local))
            val_dataset = val_dataset.shuffle(buffer_size=60000).batch(batch_size)

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
               
                if step==0 and epoch == 0:
                  prev_grad = concat_grads
                else:
                  grad_var_l2(tf.norm(concat_grads - prev_grad, ord=2)**2/tf.norm(prev_grad, ord=2)**2)
                  grad_var_l1(tf.norm(concat_grads - prev_grad, ord=1)/tf.norm(prev_grad, ord=1))
                  prev_grad = concat_grads

                # Compute top-k of grad/u
                grad_tx = grad
                if topk:
                    k_epoch = k
                    n_epoch = 20
                    if k_decay == "lin":
                        if epoch <= n_epoch:
                            k_epoch = int(k - math.floor((epoch)*(k-kf)/(num_epoch)))
                        else:
                            k_epoch = kf
                    elif k_decay == "exp":
                        k_epoch = int(k*math.pow(0.8, epoch) + kf)
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
                    np_top = np.array(grad_tx)
                    q = Quantize()
                    q.bitwidth = quant
                    q_np_top = []

                    for each_idx in range(len(np_top)):
                        q_w = q.quantize(np_top[each_idx])
                        q_np_top.append(q_w)
                        if fbk:
                            # Feedback error correction (overwrite from before)
                            r[each_idx] = u[each_idx] - q_np_top[each_idx]
                    
                    grad_tx = q_np_top
                        
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
                #optimizer.learning_rate = alpha
                # schedule
                alpha_5 = 0.1
                if epoch < 4:
                    # ramp up
                    optimizer.learning_rate = alpha + epoch*(alpha_5-alpha)/3
                elif epoch >= 4 and epoch < 8:
                    # ramp down
                    optimizer.learning_rate = alpha_5 - (epoch-4)*(alpha_5-alpha)/3
                else:
                    optimizer.learning_rate = alpha
            #optimizer.momentum = 0.9
            optimizer.apply_gradients(zip(grad_rx, model.trainable_weights))

        print(f"Accuracy over epoch {train_accuracy.result()}")
        print(f"Loss over epoch {train_loss.result()}")
        
        if rank == 3:
            # Log train metrics
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
            #train_acc = train_accuracy.result()
            print(f"Accuracy over epoch {train_acc}")

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

            # Log  metric
            with grad_var_l2_writer.as_default():
                tf.summary.scalar('grad_var_l2', grad_var_l2.result(), step=epoch)

            with grad_var_l1_writer.as_default():
                tf.summary.scalar('grad_var_l1', grad_var_l1.result(), step=epoch)

        val_loss.reset_states()
        val_accuracy.reset_states()
        grad_var_l2.reset_states()
        grad_var_l1.reset_states()
        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
