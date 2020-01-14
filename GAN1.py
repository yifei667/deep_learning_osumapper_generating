import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re, subprocess, json
from datetime import datetime
from tensorflow import keras
import sys

from tfhelper import *
from plthelper import MyLine, plot_history
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper

GAN_PARAMS = {
    "divisor" : 4,
    "good_epoch" : 6,
    "max_epoch" : 25,
    "note_group_size" : 10,
    "g_epochs" : 3,
    "c_epochs" : 7,
    "g_batch" : 50,
    "g_input_size" : 50,
    "c_true_batch" : 50,
    "c_false_batch" : 10,
    "slider_max_ticks" : 8,
    "next_from_slider_end" : False
}

def read_npz(fn):
    with np.load(fn) as data:
        objs = data["objs"]
        obj_indices = [i for i,k in enumerate(objs) if k == 1]
        predictions = data["predictions"]
        momenta = data["momenta"]
        ticks = data["ticks"]
        timestamps = data["timestamps"]
        sv = data["sv"]
        dist_multiplier = data["dist_multiplier"]
    return objs, obj_indices, predictions, momenta, ticks, timestamps, sv, dist_multiplier

rhythm_fn = sys.argv[1]
flow_fn = sys.argv[2]

# unfiltered_objs, obj_indices, unfiltered_predictions, unfiltered_momenta, unfiltered_ticks, unfiltered_timestamps, unfiltered_sv, dist_multiplier = read_npz("rhythm_data_1.npz")
unfiltered_objs, obj_indices, unfiltered_predictions, unfiltered_momenta, unfiltered_ticks, unfiltered_timestamps, unfiltered_sv, dist_multiplier = read_npz(rhythm_fn)


first_step_objs =        unfiltered_objs[obj_indices]
first_step_predictions = unfiltered_predictions[obj_indices]
first_step_momenta =     unfiltered_momenta[obj_indices]
first_step_ticks =       unfiltered_ticks[obj_indices]
first_step_timestamps =  unfiltered_timestamps[obj_indices]
first_step_sv =          unfiltered_sv[obj_indices]

momentum_multiplier = 1.0
angular_momentum_multiplier = 1.0

first_step_is_slider = first_step_predictions[:, 2]
first_step_is_spinner = first_step_predictions[:, 3]
first_step_is_sliding = first_step_predictions[:, 4]
first_step_is_spinning = first_step_predictions[:, 5]

skip_this = False
new_obj_indices = []
slider_ticks = []
slider_max_ticks = GAN_PARAMS["slider_max_ticks"]
for i in range(len(first_step_objs)):
    if skip_this:
        first_step_is_slider[i] = 0
        skip_this = False
        continue
    if first_step_is_slider[i]: # this one is a slider!!
        if i == first_step_objs.shape[0]-1: # Last Note.
            new_obj_indices.append(i)
            slider_ticks.append(slider_max_ticks)
            continue
        if first_step_ticks[i+1] >= first_step_ticks[i] + slider_max_ticks + 1: # too long! end here
            new_obj_indices.append(i)
            slider_ticks.append(slider_max_ticks)
        else:
            skip_this = True
            new_obj_indices.append(i)
            slider_ticks.append(max(1, first_step_ticks[i+1] - first_step_ticks[i]))
    else: # not a slider!
        new_obj_indices.append(i)
        slider_ticks.append(0)

# Filter the removed objects out!
objs =        first_step_objs[new_obj_indices]
predictions = first_step_predictions[new_obj_indices]
momenta =     first_step_momenta[new_obj_indices]
ticks =       first_step_ticks[new_obj_indices]
timestamps =  first_step_timestamps[new_obj_indices]
is_slider =   first_step_is_slider[new_obj_indices]
is_spinner =  first_step_is_spinner[new_obj_indices]
is_sliding =  first_step_is_sliding[new_obj_indices]
is_spinning = first_step_is_spinning[new_obj_indices]
sv =          first_step_sv[new_obj_indices]
slider_ticks = np.array(slider_ticks)

# get divisor from GAN_PARAMS
divisor = GAN_PARAMS["divisor"]

# should be slider length each tick, which is usually SV * SMP * 100 / 4
# e.g. SV 1.6, timing section x1.00, 1/4 divisor, then slider_length_base = 40
slider_length_base = sv // divisor

# these data must be kept consistent with the sliderTypes in load_map.js
slider_types = np.random.randint(0, 5, is_slider.shape).astype(int) # needs to determine the slider types!! also it is 5!!!
slider_type_rotation = np.array([0, -0.40703540572409336, 0.40703540572409336, -0.20131710837464062, 0.20131710837464062])
slider_cos = np.cos(slider_type_rotation)
slider_sin = np.sin(slider_type_rotation)

slider_cos_each = slider_cos[slider_types]
slider_sin_each = slider_sin[slider_types]

# this is vector length! I should change the variable name probably...
slider_type_length = np.array([1.0, 0.97, 0.97, 0.97, 0.97])

slider_lengths = np.array([slider_type_length[int(k)] * slider_length_base[i] for i, k in enumerate(slider_types)]) * slider_ticks

tick_lengths_pre = (timestamps[1:] - timestamps[:-1]) / (ticks[1:] - ticks[:-1])
tick_lengths = np.concatenate([tick_lengths_pre, [tick_lengths_pre[-1]]])
timestamps_note_end = timestamps + slider_ticks * tick_lengths

timestamps_plus_1 = np.concatenate([timestamps[1:], timestamps[-1:] + (timestamps[-1:] - timestamps[-2:-1])])

if GAN_PARAMS["next_from_slider_end"]:
    timestamps_after = timestamps_plus_1 - timestamps_note_end
    timestamps_before = np.concatenate([[6662], timestamps_after[:-1]]) # why 6662????
    note_distances = timestamps_before * momenta[:, 0] * momentum_multiplier
else:
    timestamps_after = timestamps_plus_1 - timestamps
    timestamps_before = np.concatenate([[4777], timestamps_after[:-1]]) # why 4777????
    note_distances = timestamps_before * momenta[:, 0] * momentum_multiplier
note_angles = timestamps_before * momenta[:, 1] * angular_momentum_multiplier

is_slider = predictions[:, 2]
is_sliding = predictions[:, 4]

root = "mapdata/"

chunk_size = GAN_PARAMS["note_group_size"]
step_size = 5

max_x = 512
max_y = 384

# "TICK", "TIME", "TYPE", "X", "Y", "IN_DX", "IN_DY", "OUT_DX", "OUT_DY"
def read_map_npz(file_path):
    with np.load(file_path) as data:
        flow_data = data["flow"]
    return flow_data

# TICK, TIME, TYPE, X, Y, IN_DX, IN_DY, OUT_DX, OUT_DY
def read_maps():
    result = []
    for file in os.listdir(root):
        if file.endswith(".npz"):
            #print(os.path.join(root, file))
            flow_data = read_map_npz(os.path.join(root, file))
            for i in range(0, (flow_data.shape[0] - chunk_size) // step_size):
                chunk = flow_data[i * step_size:i * step_size + chunk_size]
                result.append(chunk)
                
    # normalize the TICK col and remove TIME col
    result = np.array(result)[:, :, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    result[:, :, 0] %= divisor
    result[:, :, 2] /= max_x
    result[:, :, 3] /= max_y
    result[:, :, 8] /= max_x
    result[:, :, 9] /= max_y
    
    # TICK, TYPE, X, Y, IN_DX, IN_DY, OUT_DX, OUT_DY, END_X, END_Y
    # only use X,Y,OUT_DX,OUT_DY,END_X,END_Y
    result = np.array(result)[:, :, [2, 3, 6, 7, 8, 9]]
    return result

with np.load(flow_fn) as flow_dataset:
        maps = flow_dataset["maps"]
        labels = np.ones(maps.shape[0])

order2 = np.argsort(np.random.random(maps.shape[0]))
special_train_data = maps[order2]
special_train_labels = labels[order2]

def build_classifier_model():
    model = keras.Sequential([
        # keras.layers.SimpleRNN(64, input_shape=(special_train_data.shape[1], special_train_data.shape[2])),
        keras.layers.GRU(64,input_shape=(special_train_data.shape[1], special_train_data.shape[2])),
        keras.layers.Dense(128),# activation=tf.nn.elu, input_shape=(train_data.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation=tf.nn.tanh),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation=tf.nn.tanh),
        keras.layers.Lambda(lambda x: (x+1)/2, output_shape=(1,)),
    ])
    
    try:
        optimizer = tf.optimizers.Adagrad(0.01) #Adamoptimizer?
    except:
        optimizer = tf.train.AdagradOptimizer(0.01) #Adamoptimizer?

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[keras.metrics.mae])
    return model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


# A regularizer to keep the map inside the box.
# It's so the sliders and notes don't randomly fly out of the screen!
def inblock_loss(vg):
    wall_var_l = tf.where(tf.less(vg, 0.2), tf.square(0.3 - vg), 0 * vg)
    wall_var_r = tf.where(tf.greater(vg, 0.8), tf.square(vg - 0.7), 0 * vg)
    return tf.reduce_mean(tf.reduce_mean(wall_var_l + wall_var_r, axis=2), axis=1)

def inblock_trueness(vg):
    wall_var_l = tf.cast(tf.less(vg, 0), tf.float32)
    wall_var_r = tf.cast(tf.greater(vg, 1), tf.float32)
    return tf.reduce_mean(tf.reduce_mean(wall_var_l + wall_var_r, axis=2), axis=1)

def cut_map_chunks(c):
    r = []
    for i in range(0, (c.shape[0] - chunk_size) // step_size):
        chunk = c[i * step_size:i * step_size + chunk_size]
        r.append(chunk)
    return tf.stack(r)

def construct_map_with_sliders(var_tensor, extvar=[]):
    var_tensor = tf.cast(var_tensor, tf.float32)
    var_shape = var_tensor.shape
    wall_l = 0.15
    wall_r = 0.85
    x_max = 512
    y_max = 384
    out = []
    cp = tf.constant([256, 192, 0, 0])
    phase = 0
    half_tensor = var_shape[1]//4
    
    # length multiplier
    if "length_multiplier" in extvar:
        length_multiplier = extvar["length_multiplier"]
    else:
        length_multiplier = 1

    # notedists
    if "begin" in extvar:
        begin_offset = extvar["begin"]
    else:
        begin_offset = 0
    
#     note_distances_now = length_multiplier * np.expand_dims(note_distances[begin_offset:begin_offset+half_tensor], axis=0)
#     note_angles_now = np.expand_dims(note_angles[begin_offset:begin_offset+half_tensor], axis=0)
    
    relevant_tensors = extvar["relevant_tensors"]
    relevant_is_slider =      relevant_tensors["is_slider"]
    relevant_slider_lengths = relevant_tensors["slider_lengths"]
    relevant_slider_types =   relevant_tensors["slider_types"]
    relevant_slider_cos =     relevant_tensors["slider_cos_each"]
    relevant_slider_sin =     relevant_tensors["slider_sin_each"]
    relevant_note_distances = relevant_tensors["note_distances"]
    relevant_note_angles =    relevant_tensors["note_angles"]
    
    note_distances_now = length_multiplier * tf.expand_dims(relevant_note_distances, axis=0)
    note_angles_now = tf.expand_dims(relevant_note_angles, axis=0)

    # init
    l = tf.convert_to_tensor(note_distances_now, dtype="float32")
    sl = l * 0.7
    sr = tf.convert_to_tensor(note_angles_now, dtype="float32")
    
    cos_list = var_tensor[:, 0:half_tensor * 2]
    sin_list = var_tensor[:, half_tensor * 2:]
    len_list = tf.sqrt(tf.square(cos_list) + tf.square(sin_list))
    cos_list = cos_list / len_list
    sin_list = sin_list / len_list
    
    wall_l = 0.05 * x_max + l * 0.5
    wall_r = 0.95 * x_max - l * 0.5
    wall_t = 0.05 * y_max + l * 0.5
    wall_b = 0.95 * y_max - l * 0.5
    rerand = tf.cast(tf.greater(l, y_max / 2), tf.float32)
    not_rerand = tf.cast(tf.less_equal(l, y_max / 2), tf.float32)
    
    next_from_slider_end = extvar["next_from_slider_end"]

    # generate
    if "start_pos" in extvar:
        _pre_px = extvar["start_pos"][0]
        _pre_py = extvar["start_pos"][1]
        _px = tf.cast(_pre_px, tf.float32)
        _py = tf.cast(_pre_py, tf.float32)
    else:
        _px = tf.cast(256, tf.float32)
        _py = tf.cast(192, tf.float32)
    
    # this is not important since the first position starts at _ppos + Δpos
    _x = tf.cast(256, tf.float32)
    _y = tf.cast(192, tf.float32)
    
    outputs = tf.TensorArray(tf.float32, half_tensor)

    for k in range(half_tensor):
        # r_max = 192, r = 192 * k, theta = k * 10
        rerand_x = 256 + 256 * var_tensor[:, k]
        rerand_y = 192 + 192 * var_tensor[:, k + half_tensor*2]

        delta_value_x = l[:, k] * cos_list[:, k]
        delta_value_y = l[:, k] * sin_list[:, k]

        # It is tensor calculation batched 8~32 each call, so if/else do not work here.
        wall_value_l =    tf.cast(tf.less(_px, wall_l[:, k]), tf.float32)
        wall_value_r =    tf.cast(tf.greater(_px, wall_r[:, k]), tf.float32)
        wall_value_xmid = tf.cast(tf.greater(_px, wall_l[:, k]), tf.float32) * tf.cast(tf.less(_px, wall_r[:, k]), tf.float32)
        wall_value_t =    tf.cast(tf.less(_py, wall_t[:, k]), tf.float32)
        wall_value_b =    tf.cast(tf.greater(_py, wall_b[:, k]), tf.float32)
        wall_value_ymid = tf.cast(tf.greater(_py, wall_t[:, k]), tf.float32) * tf.cast(tf.less(_py, wall_b[:, k]), tf.float32)

        x_delta = tf.abs(delta_value_x) * wall_value_l - tf.abs(delta_value_x) * wall_value_r + delta_value_x * wall_value_xmid
        y_delta = tf.abs(delta_value_y) * wall_value_t - tf.abs(delta_value_y) * wall_value_b + delta_value_y * wall_value_ymid

        _x = rerand[:, k] * rerand_x + not_rerand[:, k] * (_px + x_delta)
        _y = rerand[:, k] * rerand_y + not_rerand[:, k] * (_py + y_delta)
#         _x = _px + x_delta
#         _y = _py + y_delta
        
        # calculate output vector
        
        # slider part
        sln = relevant_slider_lengths[k]
        slider_type = relevant_slider_types[k]
        scos = relevant_slider_cos[k]
        ssin = relevant_slider_sin[k]
        _a = cos_list[:, k + half_tensor]
        _b = sin_list[:, k + half_tensor]
        # cos(a+θ) = cosa cosθ - sina sinθ
        # sin(a+θ) = cosa sinθ + sina cosθ
        _oa = _a * scos - _b * ssin
        _ob = _a * ssin + _b * scos
        cp_slider = tf.transpose(tf.stack([_x / x_max, _y / y_max, _oa, _ob, (_x + _a * sln) / x_max, (_y + _b * sln) / y_max]))
        _px_slider = tf.cond(next_from_slider_end, lambda: _x + _a * sln, lambda: _x)
        _py_slider = tf.cond(next_from_slider_end, lambda: _y + _b * sln, lambda: _y)
        
        # circle part
        _a = rerand[:, k] * cos_list[:, k + half_tensor] + not_rerand[:, k] * cos_list[:, k]
        _b = rerand[:, k] * sin_list[:, k + half_tensor] + not_rerand[:, k] * sin_list[:, k]
        cp_circle = tf.transpose(tf.stack([_x / x_max, _y / y_max, _a, _b, _x / x_max, _y / y_max]))
        _px_circle = _x
        _py_circle = _y
        
        outputs = outputs.write(k, tf.where(relevant_is_slider[k], cp_slider, cp_circle))
        _px = tf.where(tf.cast(relevant_is_slider[k], tf.bool), _px_slider, _px_circle)
        _py = tf.where(tf.cast(relevant_is_slider[k], tf.bool), _py_slider, _py_circle)

    return tf.transpose(outputs.stack(), [1, 0, 2])

class GenerativeCustomLoss(LossFunctionWrapper):
    def __init__(self,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='generative_custom_loss'):
        
        def loss_function_for_generative_model(y_true, y_pred):
            classification = y_pred
            loss1 = 1 - tf.reduce_mean(classification, axis=1)
            return loss1
        
        super(GenerativeCustomLoss, self).__init__(loss_function_for_generative_model, name=name, reduction=reduction)

class BoxCustomLoss(LossFunctionWrapper):
    def __init__(self,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='generative_custom_loss'):
        
        def box_loss(y_true, y_pred):
            map_part = y_pred
            return inblock_loss(map_part[:, :, 0:2]) + inblock_loss(map_part[:, :, 4:6])
        
        super(BoxCustomLoss, self).__init__(box_loss, name=name, reduction=reduction)

class AlwaysZeroCustomLoss(LossFunctionWrapper): # why does TF not include this! this is very important in certain situations
    def __init__(self,
        reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
        name='generative_custom_loss'):
        
        def alw_zero(y_true, y_pred):
            return tf.convert_to_tensor(0, dtype=tf.float32)
        
        super(AlwaysZeroCustomLoss, self).__init__(alw_zero, name=name, reduction=reduction)
        
        
class KerasCustomMappingLayer(keras.layers.Layer):
    def __init__(self, extvar, output_shape=(special_train_data.shape[1], special_train_data.shape[2]), *args, **kwargs):
        self.extvar = extvar
        self._output_shape = output_shape
        self.extvar_begin = tf.Variable(tf.convert_to_tensor(extvar["begin"], dtype=tf.int32), trainable=False)
        self.extvar_lmul =  tf.Variable(tf.convert_to_tensor([extvar["length_multiplier"]], dtype=tf.float32), trainable=False)
        self.extvar_nfse =  tf.Variable(tf.convert_to_tensor(extvar["next_from_slider_end"], dtype=tf.bool), trainable=False)
        self.note_group_size = GAN_PARAMS["note_group_size"]
        
        self.extvar_spos =  tf.Variable(tf.cast(tf.zeros((2, )), tf.float32), trainable=False)
        self.extvar_rel =   tf.Variable(tf.cast(tf.zeros((7, self.note_group_size)), tf.float32), trainable=False)
        
        super(KerasCustomMappingLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape): # since this is a static layer, no building is required
        pass
    
    def set_extvar(self, extvar):
        self.extvar = extvar
        
        # Populate extvar with the rel variable (this will modify the input extvar)
        begin_offset = extvar["begin"]
        self.extvar["relevant_tensors"] = {
            "is_slider"       : tf.convert_to_tensor(is_slider      [begin_offset : begin_offset + self.note_group_size], dtype=tf.bool),
            "slider_lengths"  : tf.convert_to_tensor(slider_lengths [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
            "slider_types"    : tf.convert_to_tensor(slider_types   [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
            "slider_cos_each" : tf.convert_to_tensor(slider_cos_each[begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
            "slider_sin_each" : tf.convert_to_tensor(slider_sin_each[begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
            "note_distances" :  tf.convert_to_tensor(note_distances [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32),
            "note_angles" :     tf.convert_to_tensor(note_angles    [begin_offset : begin_offset + self.note_group_size], dtype=tf.float32)
        }
        
        # Continue
        self.extvar_begin.assign(extvar["begin"])
        self.extvar_spos.assign(extvar["start_pos"])
        self.extvar_lmul.assign([extvar["length_multiplier"]])
        self.extvar_nfse.assign(extvar["next_from_slider_end"])
        self.extvar_rel.assign(tf.convert_to_tensor([
            is_slider      [begin_offset : begin_offset + self.note_group_size],
            slider_lengths [begin_offset : begin_offset + self.note_group_size],
            slider_types   [begin_offset : begin_offset + self.note_group_size],
            slider_cos_each[begin_offset : begin_offset + self.note_group_size],
            slider_sin_each[begin_offset : begin_offset + self.note_group_size],
            note_distances [begin_offset : begin_offset + self.note_group_size],
            note_angles    [begin_offset : begin_offset + self.note_group_size]
        ], dtype=tf.float32))

    # Call method will sometimes get used in graph mode,
    # training will get turned into a tensor
#     @tf.function
    def call(self, inputs, training=None):
        mapvars = inputs
        start_pos = self.extvar_spos
        rel = self.extvar_rel
        extvar = {
            "begin" : self.extvar_begin,
            # "start_pos" : self.extvar_start_pos,
            "start_pos" : tf.cast(start_pos, tf.float32),
            "length_multiplier" : self.extvar_lmul,
            "next_from_slider_end" : self.extvar_nfse,
            # "relevant_tensors" : self.extvar_rel
            "relevant_tensors" : {
                "is_slider"       : tf.cast(rel[0], tf.bool),
                "slider_lengths"  : tf.cast(rel[1], tf.float32),
                "slider_types"    : tf.cast(rel[2], tf.float32),
                "slider_cos_each" : tf.cast(rel[3], tf.float32),
                "slider_sin_each" : tf.cast(rel[4], tf.float32),
                "note_distances"  : tf.cast(rel[5], tf.float32),
                "note_angles"     : tf.cast(rel[6], tf.float32)
            }
        }
        result = construct_map_with_sliders(mapvars, extvar=extvar)
        return result
        

loss_ma = [90, 90, 90]
extvar = {"begin": 10}

def plot_current_map(inputs):
    # plot it each epoch
    mp = construct_map_with_sliders(inputs, extvar=extvar)
    # to make it clearer, add the start pos
    npa = np.concatenate([[np.concatenate([extvar["start_pos"] / np.array([512, 384]), [0, 0]])], tf.stack(mp).numpy().squeeze()])
    fig, ax = plt.subplots()
    x, y = np.transpose(npa)[0:2]
    #x, y = np.random.rand(2, 20)
    line = MyLine(x, y, mfc='red', ms=12)
    line.text.set_color('red')
    line.text.set_fontsize(16)
    ax.add_line(line)
    plt.show()

def generative_model(in_params, out_params, loss_func='mse'):
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(in_params,)),# activation=tf.nn.elu, input_shape=(train_data.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation=tf.nn.tanh),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(out_params, activation=tf.nn.tanh)#,
#         keras.layers.Lambda(lambda x: (x+1)/2, output_shape=(out_params,))
    ])

    try:
        optimizer = tf.optimizers.Adam(0.002) #Adamoptimizer?
    except:
        optimizer = tf.train.AdamOptimizer(0.002) #Adamoptimizer?

    model.compile(loss=loss_func,
                optimizer=optimizer,
                metrics=[keras.metrics.mae])
    return model

def mixed_model(generator, mapping_layer, discriminator, in_params):
    note_group_size = GAN_PARAMS["note_group_size"]
    inp = keras.layers.Input(shape=(in_params,))
    start_pos = keras.layers.Input(shape = (2,))#tf.convert_to_tensor([0, 0], dtype=tf.float32)
    rel = keras.layers.Input(shape = (7, note_group_size))#tf.zeros((5, note_group_size), dtype=tf.float32)
    interm1 = generator(inp)
    interm2 = mapping_layer(interm1)
    end = discriminator(interm2)
    model = keras.Model(inputs = inp, outputs = [interm1, interm2, end])
    
    discriminator.trainable = False

    try:
        optimizer = tf.optimizers.Adam(0.001) #Adamoptimizer?
    except:
        optimizer = tf.train.AdamOptimizer(0.001) #Adamoptimizer?
        
    losses = [AlwaysZeroCustomLoss(), BoxCustomLoss(), GenerativeCustomLoss()]

    model.compile(loss=losses,
                  loss_weights=[1e-8, 1, 1],
                optimizer=optimizer)
    return model

def conv_input(inp, extvar):
#     Now it only uses single input
    return inp


plot_noise = np.random.random((1, GAN_PARAMS["g_input_size"]))

# Pre-fit classifier for 1 epoch
# history = classifier_model.fit(actual_train_data, actual_train_labels, epochs=1,
#                     validation_split=0.2, verbose=0,
#                     callbacks=[])

# build models first, then train (it is faster in TF 2.0)

def make_models():
        
    extvar["begin"] = 0
    extvar["start_pos"] = [256, 192]
    extvar["length_multiplier"] = 1
    extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"]
    
    classifier_model = build_classifier_model()
    note_group_size = GAN_PARAMS["note_group_size"]
    g_input_size = GAN_PARAMS["g_input_size"]
    
    gmodel = generative_model(g_input_size, note_group_size * 4)
    mapping_layer = KerasCustomMappingLayer(extvar)
    mmodel = mixed_model(gmodel, mapping_layer, classifier_model, g_input_size)
    
    default_weights = mmodel.get_weights()
    
    return gmodel, mapping_layer, classifier_model, mmodel, default_weights

def set_extvar(models, extvar):
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models
    mapping_layer.set_extvar(extvar)
    
def reset_model_weights(models):
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models
    weights = default_weights
    mmodel.set_weights(weights)

# we can train all the classifiers first, onto Epoch X [x = 1~10]
# then train the generators to fit to them
# to reduce some training time.
# but i think it doesn't work too well since it's the generator which is slow...d
def generate_set(models, begin = 0, start_pos=[256, 192], group_id=-1, length_multiplier=1, plot_map=True):
    extvar["begin"] = begin
    extvar["start_pos"] = start_pos
    extvar["length_multiplier"] = length_multiplier
    extvar["next_from_slider_end"] = GAN_PARAMS["next_from_slider_end"]
    
    note_group_size = GAN_PARAMS["note_group_size"]
    max_epoch = GAN_PARAMS["max_epoch"]
    good_epoch = GAN_PARAMS["good_epoch"] - 1
    g_multiplier = GAN_PARAMS["g_epochs"]
    c_multiplier = GAN_PARAMS["c_epochs"]
    g_batch = GAN_PARAMS["g_batch"]
    g_input_size = GAN_PARAMS["g_input_size"]
    c_true_batch = GAN_PARAMS["c_true_batch"]
    c_false_batch = GAN_PARAMS["c_false_batch"]
    
    reset_model_weights(models)
    set_extvar(models, extvar)
    gmodel, mapping_layer, classifier_model, mmodel, default_weights = models
    
    # see the summaries
#     gmodel.summary()
#     classifier_model.summary()
#     mmodel.summary()
    g_losses = []
    c_losses = []
    for i in range(max_epoch):
        
        gnoise = np.random.random((g_batch, g_input_size))
        glabel = [np.zeros((g_batch, note_group_size * 4)), np.ones((g_batch,)), np.ones((g_batch,))]
        ginput = conv_input(gnoise, extvar)
        
        # fit mmodel instead of gmodel
        history = mmodel.fit(ginput, glabel, epochs=g_multiplier,
                            validation_split=0.2, verbose=0,
                            callbacks=[])
        
        pred_noise = np.random.random((c_false_batch, g_input_size))
        pred_input = conv_input(pred_noise, extvar)
        predicted_maps_data, predicted_maps_mapped, _predclass = mmodel.predict(pred_input)
        new_false_maps = predicted_maps_mapped
        new_false_labels = np.zeros(c_false_batch)
        

        rn = np.random.randint(0, special_train_data.shape[0], (c_true_batch,))
        actual_train_data = np.concatenate((new_false_maps, special_train_data[rn]), axis=0) #special_false_data[st:se], 
        actual_train_labels = np.concatenate((new_false_labels, special_train_labels[rn]), axis=0) #special_false_labels[st:se], 
        
    
        history2 = classifier_model.fit(actual_train_data, actual_train_labels, epochs=c_multiplier,
                            validation_split=0.2, verbose=0,
                            callbacks=[])
        
        # calculate the losses
        g_loss = np.mean(history.history['loss'])
        c_loss = np.mean(history2.history['loss'])
        print("Group {}, Epoch {}: G loss: {} vs. C loss: {}".format(group_id, 1+i, g_loss, c_loss))
        g_losses.append(g_loss)
        c_losses.append(c_loss)
        # delete the history to free memory
        del history, history2
        
        # make a new set of notes
        res_noise = np.random.random((1, g_input_size))
        res_input = conv_input(res_noise, extvar)
        _resgenerated, res_map, _resclass = mmodel.predict(res_input)
        if plot_map:
            plot_current_map(tf.convert_to_tensor(res_map, dtype=tf.float32))
        
        # early return if found a good solution
        # good is (inside the map boundary)
        if i >= good_epoch:
#             current_map = construct_map_with_sliders(tf.convert_to_tensor(res, dtype="float32"), extvar=extvar)
            current_map = res_map
            if inblock_trueness(current_map[:, :, 0:2]).numpy()[0] == 0 and inblock_trueness(current_map[:, :, 4:6]).numpy()[0] == 0:
                # debugging options to check map integrity
#                 print(tf.reduce_mean(current_map))
#                 print("-----MAPLAYER-----")
#                 print(tf.reduce_mean(mapping_layer(conv_input(tf.convert_to_tensor(_resgenerated, dtype="float32"), extvar))))
#                 print("-----CMWS-----")
#                 print(tf.reduce_mean(construct_map_with_sliders(tf.convert_to_tensor(_resgenerated, dtype="float32"), extvar=mapping_layer.extvar)))
                break

    # plot_history(history)
    # plot_history(history2)


    if plot_map:
        for i in range(3): # from our testing, any random input generates nearly the same map
            plot_noise = np.random.random((1, g_input_size))
            plot_input = conv_input(plot_noise, extvar)
            _plotgenerated, plot_mapped, _plotclass = mmodel.predict(plot_input)
            plot_current_map(tf.convert_to_tensor(plot_mapped, dtype=tf.float32))
    

    
    return res_map.squeeze(), g_losses, c_losses

# generate the map (main function)
# dist_multiplier in #6 is used here
def generate_map():
    o = []
    note_group_size = GAN_PARAMS["note_group_size"]
    pos = [np.random.randint(100, 412), np.random.randint(80, 304)]
    models = make_models()
    g_l = []
    c_l = []
    print("# of groups: {}".format(timestamps.shape[0] // note_group_size))
    for i in range(timestamps.shape[0] // note_group_size):
        z, g_losses, c_losses = generate_set(models, begin = i * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = i, plot_map=False) 
        z *= np.array([512, 384, 1, 1, 512, 384])
        # g_l += g_losses
        # c_l += c_losses
        pos = z[-1, 0:2]
        o.append(z)
    a = np.concatenate(o, axis=0)

    # Print last plot
    plt.plot(g_losses, "b")
    plt.plot(c_losses, "y")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Generator', 'Discriminator'], loc='upper left')
    plt.show()
    plt.savefig("GAN_losses.png")
    return a

# generate a test map (debugging function)
# dist_multiplier in #6 is used here
def generate_test():
    o = []
    pos = [384, 288]
    note_group_size = GAN_PARAMS["note_group_size"]
    generate_set(begin = 3 * note_group_size, start_pos = pos, length_multiplier = dist_multiplier, group_id = 3, plot_map=True)

# for debugging only! it should be sent to node load_map.js c instead.
def print_osu_text(a):
    for i, ai in enumerate(a):
        if not is_slider[i]:
            print("{},{},{},1,0,0:0:0".format(int(ai[0]), int(ai[1]), int(timestamps[i])))
        else:
            print("{},{},{},2,0,L|{}:{},1,{},0:0:0".format(int(ai[0]), int(ai[1]), int(timestamps[i]), int(round(ai[0] + ai[2] * slider_lengths[i])), int(round(ai[1] + ai[3] * slider_lengths[i])), int(slider_length_base[i] * slider_ticks[i])))
    
osu_a = generate_map()

def convert_to_osu_obj(obj_array):
    output = []
    for i, obj in enumerate(obj_array):
        if not is_slider[i]: # is a circle does not consider spinner for now.
            obj_dict = {
                "x": int(obj[0]),
                "y": int(obj[1]),
                "type": 1,
                "time": int(timestamps[i]),
                "hitsounds": 0,
                "extHitsounds": "0:0:0",
                "index": i
            }
        else:
            obj_dict = {
                "x": int(obj[0]),
                "y": int(obj[1]),
                "type": 2,
                "time": int(timestamps[i]),
                "hitsounds": 0,
                "extHitsounds": "0:0:0",
                "sliderGenerator": {
                    "type": int(slider_types[i]),
                    "dOut": [float(obj[2]), float(obj[3])],
                    "len": float(slider_length_base[i] * slider_ticks[i]),
                    "ticks": int(slider_ticks[i]),
                    "endpoint": [int(obj[4]), int(obj[5])]
                },
                "index": i
            }
        output.append(obj_dict)
    return output

def get_osu_file_name(metadata):
    artist = metadata["artist"]
    title = metadata["title"]
    creator = metadata["creator"]
    diffname = metadata["diffname"]
    outname = (artist+" - " if len(artist) > 0 else "") + title + " (" + creator + ") [" + diffname + "].osu"
    outname = re.sub("[^a-zA-Z0-9\(\)\[\] \.\,\!\~\`\{\}\-\_\=\+\&\^\@\#\$\%\\']","", outname)
    return outname

osu_obj_array = convert_to_osu_obj(osu_a)

with open("mapthis.json", encoding="utf-8") as map_json:
    map_dict = json.load(map_json)
    map_meta = map_dict["meta"]
    filename = get_osu_file_name(map_meta)
    map_dict["obj"] = osu_obj_array

with open('mapthis.json', 'w', encoding="utf-8") as outfile:
    json.dump(map_dict, outfile, ensure_ascii=False)

subprocess.call(["node", "load_map.js", "c", "mapthis.json", filename])
print("success! finished on: {}".format(datetime.now()))