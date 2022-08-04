# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""

import numpy as np
import tensorflow as tf 
import os 
import shutil
import random
import math
import scipy.io as sio
import time
from skimage import measure
# import binvox_rw
import argparse
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.utils.libkdtree import KDTree
import re


parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="026911156")
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--dataset', type=str, default="shapenet")
parser.add_argument('--test_type', type=int, default=1)
parser.add_argument('--thresh', type=float, default=0.005)
parser.add_argument('--pattern_num', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_local', type=int, default=None)
parser.add_argument('--gauss_value', type=float, default=1.0)
parser.add_argument('--start_index', type=int, default=1)
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx


BS = a.batch_size
POINT_NUM = 5000
POINT_NUM_GT = 20000
PATTERN_NUM = a.pattern_num
TEST_TYPE = a.test_type
THRESH = a.thresh
GAUSS_VALUE = a.gauss_value
START_INDEX = a.start_index
TEST_LOCAL = a.test_local
PATTERN_POINT_NUM = int(POINT_NUM / PATTERN_NUM)
INPUT_DIR = os.path.join(a.data_dir, 'samples_' + a.class_idx)
OUTPUT_DIR = a.out_dir
if(a.dataset=="shapenet" or a.dataset=="other"):
    GT_DIR = '/data/mabaorui/project4/data/ShapeNet/' + a.class_idx + '/'
if(a.dataset=="famous"):
    GT_DIR = './data/famous_noisefree/03_meshes/'

TRAIN = a.train
bd = 0.55

if(TRAIN):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)
else:
    BS = 1


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

#        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
#        # Handle normals that point into wrong direction gracefully
#        # (mostly due to mehtod not caring about this in generation)
#        normals_dot_product = np.abs(normals_dot_product)
        
        normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
        normals_dot_product = normals_dot_product.sum(axis=-1)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty


        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        #print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        print('chamferL2:',chamferL2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('normals_correctness:',normals_correctness,'chamferL1:',chamferL1)
        return normals_correctness, chamferL1, chamferL2


def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)


def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)


def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)
    

def chamfer_distance_tf_None(array1, array2):
    array1 = tf.reshape(array1,[-1,3])
    array2 = tf.reshape(array2,[-1,3])
    av_dist1 = av_dist_None(array1, array2)
    av_dist2 = av_dist_None(array2, array1)
    return av_dist1+av_dist2


def distance_matrix_None(array1, array2, num_point, num_features = 3):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances


def chamfer_distance_tf_None2(p, q):
    from nn_distance import tf_nndistance
    a,b,c,d = tf_nndistance.nn_distance(p, q)
    cd1 = tf.reduce_mean(a)
    cd2 = tf.reduce_mean(c)
    return cd1, cd2


def chamfer_distance_tf_None3(p, q): 
    from nn_distance import tf_nndistance
    a,b,c,d = tf_nndistance.nn_distance(p, q)
    cd = tf.reduce_mean(a) 
    return cd


def distance(pred, gt):
    pred = pred[:, None, :, :]
    gt = gt[:, :, None, :]
    diff = (pred - gt)**2
    dist = tf.sqrt(tf.reduce_sum(diff, 3) + 1e-16)
    return dist


def av_dist_None(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix_None(array1, array2,points_input_num[0,0])
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances


def test(sess, index, test_local_id, input_thresh):
        s = np.arange(-bd,bd, (2*bd)/128)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        POINT_NUM_GT_bs = np.array(vox_size).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        input_points_2d_bs = []
        for i in s:
            for j in s:
                for k in s:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        print('input_points_2d_bs',input_points_2d_bs.shape)
        input_points_2d_bs = input_points_2d_bs.reshape((vox_size,vox_size,vox_size,3))
        POINT_NUM_GT_bs = np.array(vox_size*vox_size).reshape(1,1)

        test_num = SHAPE_NUM
        print('test_num:',test_num)
        cd = 0
        nc = 0
        cd2 = 0
        for epoch in range(test_num):
            print('test:',epoch)
            
            vox = []
            mis = []
            feature_bs = []
            for j in range(vox_size*vox_size):
                t = np.zeros(SHAPE_NUM)
                t[epoch] = 1
                feature_bs.append(t)
            feature_bs = np.asarray(feature_bs)
            for i in range(vox_size):
                input_points_2d_bs_t = input_points_2d_bs[i,:,:,:]
                input_points_2d_bs_t = input_points_2d_bs_t.reshape(BS, vox_size*vox_size, 3)
                feature_bs_t = feature_bs.reshape(BS,vox_size*vox_size,SHAPE_NUM)
                cent = all_centers[epoch,:,:].reshape(BS, PATTERN_NUM, 3)
                sdf_c = sess.run([sdf],feed_dict={input_points_3d:input_points_2d_bs_t,feature:feature_bs_t,center: cent, points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
         
                vox.append(sdf_c)    
               
            vox = np.asarray(vox)
            vox = vox.reshape((vox_size,vox_size,vox_size))
            vox_max = np.max(vox.reshape((-1)))
            vox_min = np.min(vox.reshape((-1)))
            print('max_min:',vox_max,vox_min)
            
            threshs = [input_thresh]
            for thresh in threshs:
                print(np.sum(vox>thresh),np.sum(vox<thresh))
                
                if(np.sum(vox>0.0)<np.sum(vox<0.0)):
                    thresh = -thresh
                print('model:',epoch,'thresh:',thresh)
                vertices, triangles = libmcubes.marching_cubes(vox, thresh)
                if(vertices.shape[0]<10 or triangles.shape[0]<10):
                    print('no sur---')
                    continue
                if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                    triangles_t = []
                    for it in range(triangles.shape[0]):
                        tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                        triangles_t.append(tt)
                    triangles_t = np.asarray(triangles_t)
                else:
                    triangles_t = triangles
                    triangles_t = np.asarray(triangles_t)

                vertices -= 0.5
                # Undo padding
                vertices -= 1
                # Normalize to bounding box
                vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
                vertices = 1.1 * (vertices - 0.5)
                mesh = trimesh.Trimesh(vertices, triangles_t,
                               vertex_normals=None,
                               process=False)
                if test_local_id is None:
                    mesh.export(OUTPUT_DIR +  '/occn_' + files[epoch] + '_'+ str(index) + '_' + str(thresh) + '.off')
                else:
                    if not os.path.exists(OUTPUT_DIR + '/local_' + files[epoch]):
                        os.mkdir(OUTPUT_DIR + '/local_' + files[epoch])
                    mesh.export(OUTPUT_DIR +  '/local_' + files[epoch] + '/occn_' + files[epoch] + '_'+ str(index) + '_' + str(test_local_id) + '_' + str(thresh) + '.off')
    
                mesh = trimesh.Trimesh(vertices, triangles,
                                   vertex_normals=None,
                                   process=False)
                
                if(a.dataset=="shapenet" or a.dataset == 'other'):
                    ps, idx = mesh.sample(1000000, return_index=True)
                else:
                    ps, idx = mesh.sample(10000, return_index=True)
                ps = ps.astype(np.float32)
                normals_pred = mesh.face_normals[idx]
    
                if(a.dataset=="shapenet" or a.dataset == 'other'):
                    data = np.load(GT_DIR + files[epoch] + '/pointcloud.npz')
                    pointcloud = data['points']
                    normal = data['normals']
                else:
                    mesh_gt = trimesh.load(GT_DIR + files[epoch] + '.ply')
                    pointcloud, idx_gt = mesh_gt.sample(10000, return_index=True)
                    pointcloud = pointcloud.astype(np.float32)
                    normal = mesh_gt.face_normals[idx_gt]
                
                nc_t,cd_t,cd2_t = eval_pointcloud(ps,pointcloud.astype(np.float32),normals_pred.astype(np.float32),normal.astype(np.float32))
                if test_local_id is None:
                    np.savez(OUTPUT_DIR + files[epoch]+ '_'+ str(index) + '_' + str(thresh),pp = ps, np = normals_pred, p = pointcloud, n = normal, nc = nc_t, cd = cd_t, cd2 = cd2_t)
                nc = nc + nc_t
                cd = cd + cd_t
                cd2 = cd2 + cd2_t
        print('mean_nc:',nc/test_num,'mean_cd:',cd/test_num,'cd2:',cd2/test_num)
        return cd2 / test_num


files = []
files_path = []

if(a.dataset == "shapenet"):
    f = open('./data/shapenet_val.txt','r')
    for index,line in enumerate(f):
        if(line.strip().split('/')[0]==a.class_idx):
            #print(line)
            files.append(line.strip().split('/')[1])
    f.close()

if(a.dataset == "famous"):
    f = open('./data/famous_testset.txt','r')
    for index,line in enumerate(f):
        #print(line)
        files.append(line.strip('\n'))
    f.close()
    
if(a.dataset == "other"):
    files_path = []
    files = []
    dl_dir = os.path.join(a.data_dir, 'class_list', a.class_idx + '.txt')
    f = open(dl_dir, 'r')
    data = f.readlines()
    data = [d.strip('\n') for d in data]
    if (START_INDEX + 1) * 1 > len(data):
        end_index = len(data)
        if TRAIN:
            BS = end_index - START_INDEX * 1
    else:
        end_index = (START_INDEX + 1) * 1
    for k in range(START_INDEX * 1, end_index):
        files.append(data[k])
        files_path.append(os.path.join(INPUT_DIR, data[k] + '.npz'))

print(files_path)
print(files)
SHAPE_NUM = len(files_path)
print('SHAPE_NUM:',SHAPE_NUM)

pointclouds = []
samples = []
mm = 0
all_centers = []
if(TRAIN):
    for file in files_path:
        print(file)
        path_list = file.split('/')
        name = path_list[-1].split('.')[0]
        path = os.path.join(a.data_dir, 'centers_'+ str(PATTERN_NUM) + '_' + a.class_idx, name + '_centers_' + str(PATTERN_NUM) + '.txt')
        cen = np.loadtxt(path)
        all_centers.append(cen)
        load_data = np.load(file)
        point = np.asarray(load_data['sample_near']).reshape(-1,POINT_NUM,3)
        sample = np.asarray(load_data['sample']).reshape(-1,POINT_NUM,3)
        pointclouds.append(point)
        samples.append(sample)
    pointclouds = np.asarray(pointclouds)
    samples = np.asarray(samples)
    all_centers = np.asarray(all_centers)
    print('data shape:',pointclouds.shape,samples.shape, all_centers.shape)
else:
    for file in files_path:
        path_list = file.split('/')
        name = path_list[-1].split('.')[0]
        path = os.path.join(a.data_dir, 'centers_'+ str(PATTERN_NUM) + '_' + a.class_idx, name + '_centers_' + str(PATTERN_NUM) + '.txt')
        cen = np.loadtxt(path)
        all_centers.append(cen)
    all_centers = np.asarray(all_centers)


feature = tf.placeholder(tf.float32, shape=[BS, None, SHAPE_NUM])
points_target = tf.placeholder(tf.float32, shape=[BS, None, 3])
input_points_3d = tf.placeholder(tf.float32, shape=[BS, None, 3])
points_target_num = tf.placeholder(tf.int32, shape=[1, 1])
points_input_num = tf.placeholder(tf.int32, shape=[1, 1])
center = tf.placeholder(tf.float32, shape=[BS, PATTERN_NUM, 3])
feature_f = tf.nn.relu(tf.layers.dense(feature, 128))


def LPI(points_onehot, global_feature):
    points = points_onehot[:,:,:3]
    onehot = points_onehot[:,:,3:]
    onehot_learned = tf.layers.dense(onehot, 64)
    pc_one = tf.concat([points, onehot_learned, global_feature], 2)
    
    net = tf.nn.relu(tf.layers.dense(pc_one, 256))

    print('net:',net)
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        for i in range(8):
            with tf.variable_scope("resnetBlockFC_%d" % i ):
                b_initializer=tf.constant_initializer(0.0)
                w_initializer = tf.random_normal_initializer(mean=0.0,stddev=np.sqrt(2) / np.sqrt(256))
                net = tf.layers.dense(tf.nn.relu(net), 256, kernel_initializer=w_initializer,bias_initializer=b_initializer)

    b_initializer=tf.constant_initializer(-0.5)
    w_initializer = tf.random_normal_initializer(mean=2*np.sqrt(np.pi) / np.sqrt(512), stddev = 0.000001)
    print('net:',net)
    sdf = tf.layers.dense(tf.nn.relu(net),1,kernel_initializer=w_initializer,bias_initializer=b_initializer)
    print('sdf',sdf)

    grad = tf.gradients(ys=sdf, xs=points_onehot) 
    print('grad',grad)
    grad = grad[0][:, :, :3]
    normal_p_lenght = tf.expand_dims(safe_norm(grad, axis = -1),-1)
    print('normal_p_lenght',normal_p_lenght)
    grad_norm = grad/normal_p_lenght
    print('grad_norm',grad_norm)

    g_points = points_onehot[:, :, :3] - sdf * grad_norm
    return g_points, sdf, grad_norm


patch_lens = PATTERN_NUM
test_type = TEST_TYPE
test_local = TEST_LOCAL
if TRAIN:
    neural_inputs = input_points_3d 
    init_latent_code = tf.diag(tf.ones(PATTERN_NUM))
    init_latent_code = tf.layers.dense(init_latent_code, PATTERN_NUM)
    mm = tf.tile(tf.expand_dims(init_latent_code, 1), [1,POINT_NUM,1]) # [10, 5000, 10]
    mm = mm[None, :, :, :]
    points_to_centers = distance(neural_inputs, center) # latent code weight, add exp
    latent_code_weight = tf.nn.softmax(tf.exp(-points_to_centers/GAUSS_VALUE), 1) #[BS, PATTERN_NUM, POINT_NUM]
    latent_code_weight = tf.tile(latent_code_weight[:, :, :, None], [1, 1, 1, PATTERN_NUM]) #[BS, PATTERN_NUM, POINT_NUM, LATENT_CODE_DIM]
    latent_code = tf.reduce_sum(latent_code_weight*mm, 1)
    inputs = tf.concat([neural_inputs, latent_code], 2)
    with tf.variable_scope('LPI', reuse=tf.AUTO_REUSE):
        g_points, sdf, grad = LPI(inputs, feature_f)
    test_loss_points = g_points
else:
    abc = 128*128
    neural_inputs = input_points_3d
    init_latent_code = tf.diag(tf.ones(PATTERN_NUM))
    init_latent_code = tf.layers.dense(init_latent_code, PATTERN_NUM)
    mm = tf.tile(tf.expand_dims(init_latent_code, 1), [1, abc, 1])
    mm = mm[None, :, :, :]
    points_to_centers = distance(neural_inputs, center)
    latent_code_weight = tf.nn.softmax(tf.exp(-points_to_centers/GAUSS_VALUE), 1)
    latent_code_weight = tf.tile(latent_code_weight[:, :, :, None], [1, 1, 1, PATTERN_NUM])
    latent_code = tf.reduce_sum(latent_code_weight*mm, 1)
 
    if test_local is None:
        weight = latent_code
    else:
        selected_index = test_local
        min_index = tf.argmin(points_to_centers, 1)
        index_with_selected_index = tf.where(tf.equal(min_index, selected_index))
        index_with_selected_index = index_with_selected_index[:, 1:]
        index_not_with_selected_index = tf.where(tf.not_equal(min_index, selected_index))
        index_not_with_selected_index = index_not_with_selected_index[:, 1:]

        indices = tf.cast(index_with_selected_index, tf.int64)
        updates = tf.gather(latent_code[0], indices)
        updates = updates[:, 0, :]
        shape = tf.constant([abc, PATTERN_NUM], tf.int64)   
        scatter = tf.scatter_nd(indices, updates, shape)
        
        indices_not = tf.cast(index_not_with_selected_index, tf.int64)
        updates_not = 0 * tf.gather(latent_code[0], indices_not) + 9999
        updates_not = updates_not[:, 0, :]
        shape_not = tf.constant([abc, PATTERN_NUM], tf.int64) 
        scatter_not = tf.scatter_nd(indices_not, updates_not, shape_not) 
             

        weight = scatter[None, :, :] + scatter_not[None, :, :]
   
    temp = tf.concat([neural_inputs, weight], 2)
    with tf.variable_scope('LPI', reuse=tf.AUTO_REUSE):
        g_points, sdf, grad = LPI(temp, feature_f)


a_loss, b_loss = chamfer_distance_tf_None2(points_target, g_points)
loss = a_loss+b_loss
print('loss:',loss)
loss = tf.reduce_mean(loss)


t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)


config = tf.ConfigProto(allow_soft_placement=False) 
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)



with tf.Session(config=config) as sess:
    feature_bs = np.tile(np.expand_dims(np.eye(SHAPE_NUM), 1), [1, POINT_NUM, 1])
   
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        
        start_time_5000 = time.time()
        for i in range(40002):
            start_time = time.time()
            epoch_index = 1 # np.random.choice(SHAPE_NUM, SHAPE_NUM, replace = False)
            loss_i = 0
           
            for epoch in range(1):
                rt = i % samples.shape[1] #random.randint(0,samples.shape[1]-1)
                input_points_2d_bs = samples[:,rt,:,:].reshape(BS, POINT_NUM, 3)
                point_gt = pointclouds[:,rt,:,:].reshape(BS,POINT_NUM,3)
                feature_bs_t = feature_bs[:,:,:].reshape(BS,-1,SHAPE_NUM)
                cent = all_centers[:,:,:].reshape(BS, PATTERN_NUM, 3)
                _, loss_val = sess.run([loss_optim, loss],feed_dict={input_points_3d:input_points_2d_bs,points_target:point_gt,feature:feature_bs_t,center: cent, points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
                loss_i = loss_i + loss_val

            loss_i = loss_i / SHAPE_NUM
            end_time = time.time()
            if(i%10 == 0):
                if i % 5000 == 0:
                    end_time_5000 = time.time()
                    print('time: ', end_time_5000 - start_time_5000)
                    start_time_5000 = time.time()
                print('epoch:', i, 'epoch loss:', loss_i, 'time: ', end_time - start_time)
            if(i%10000 == 0):
                print('save model')
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
    else:
        print('test start')
        for i in range(0, 1):
            index =  i * 10000 + 1
            path = OUTPUT_DIR + 'model-' + str(index)
            print(path)
            if os.path.exists(path + '.index'):
                saver.restore(sess, path)
                cd = test(sess, index, test_local, THRESH)
            else:
                break
