#!/home/telepresence/catkin_ws/src/iviz_msgs/venv/bin/python2
import sys
from numpy import NaN
sys.path.append('/home/telepresence/catkin_ws/src/iviz_msgs')
import pandas as pd
from iviz_msgs.msg import XRHandState
from tf2_msgs.msg import TFMessage
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import roslib; roslib.load_manifest('visualization_marker_tutorials')
import numpy as np
import torch as T
import pickle as pk
from data_creator.data_creator import DataCreator
import quaternion
from std_msgs.msg import String
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tf



from models.gru_model import GRUModel
from models.lstm_model import LSTMModel


sliding_window = None
predict_every_n_th_sample = 1
sample_counter_L = 0
sample_counter_R = 0
gelenke = []
left_gelenke = []
right_gelenke = []
left_, right = DataCreator.get_all_column_names_seperatly()
all_columns = left_ + right[2:]

one_sample = pd.DataFrame(data=None, index=None, dtype=None, copy=False, columns=['secs'])

model = None #Do not change this line. The variable model has to be initialised
sliding_window_size = 120
model_path = 'model/model.pth'
input_dim = 366


rospy.init_node('gesture_recognition')
tf_listener = tf.TransformListener(rospy.Duration(600.0))
rviz_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
markerArray = MarkerArray()
marker_size = 0.03

def add_one_sample(px, py, pz, ow, ox, oy, oz):
    one_sample.append(px)
    one_sample.append(py)
    one_sample.append(pz)
    one_sample.append(ox)
    one_sample.append(oy)
    one_sample.append(oz)
    one_sample.append(ow)

def process_raw_hand_data_to_df(data, trans, rot, left_flag):
    l_colum_names, r_colum_names = DataCreator().get_all_column_names_seperatly()
    if left_flag:
        df = pd.DataFrame(data=None, index=None, columns=l_colum_names, dtype=None, copy=False)
    else:
        df = pd.DataFrame(data=None, index=None, columns=r_colum_names, dtype=None, copy=False)

    one_sample = []

    one_sample.extend([data.header.stamp.secs, data.header.stamp.nsecs, data.is_valid])

    t = np.quaternion(0, trans[0], trans[1], trans[2])
    q = np.quaternion(rot[3], rot[0],rot[1], rot[2])

    q_i = q.inverse()
    basis_change_t = q_i * -t * q_i.conjugate()
    basis_change_q = q_i

    def single_basis_change(basis_change_q, basis_change_t, pt_x, pt_y, pt_z, pq_x, pq_y, pq_z, pq_w):
        pt = np.quaternion(0, pt_x, pt_y, pt_z)
        pq = np.quaternion(pq_w, pq_x, pq_y, pq_z)
        pt = basis_change_q * pt * basis_change_q.conjugate() + basis_change_t
        pq = basis_change_q * pq
        return [pt.x, pt.y, pt.z, pq.x, pq.y, pq.z, pq.w]

    one_sample = one_sample + single_basis_change(basis_change_q, basis_change_t, data.palm.translation.x, data.palm.translation.y, data.palm.translation.z,
        data.palm.rotation.x, data.palm.rotation.y, data.palm.rotation.z, data.palm.rotation.w)
    

    for i in range(len(data.thumb)):
        one_sample = one_sample + single_basis_change(basis_change_q, basis_change_t, data.thumb[i].translation.x, data.thumb[i].translation.y, data.thumb[i].translation.z,
        data.thumb[i].rotation.x, data.thumb[i].rotation.y, data.thumb[i].rotation.z, data.thumb[i].rotation.w)

    for i in range(len(data.index)):
        one_sample = one_sample + single_basis_change(basis_change_q, basis_change_t, data.index[i].translation.x, data.index[i].translation.y, data.index[i].translation.z,
        data.index[i].rotation.x, data.index[i].rotation.y, data.index[i].rotation.z, data.index[i].rotation.w)

    for i in range(len(data.middle)):
        one_sample = one_sample + single_basis_change(basis_change_q, basis_change_t, data.middle[i].translation.x, data.middle[i].translation.y, data.middle[i].translation.z,
        data.middle[i].rotation.x, data.middle[i].rotation.y, data.middle[i].rotation.z, data.middle[i].rotation.w)

    for i in range(len(data.ring)):
        one_sample = one_sample + single_basis_change(basis_change_q, basis_change_t, data.ring[i].translation.x, data.ring[i].translation.y, data.ring[i].translation.z,
        data.ring[i].rotation.x, data.ring[i].rotation.y, data.ring[i].rotation.z, data.ring[i].rotation.w)

    for i in range(len(data.little)):
        one_sample = one_sample + single_basis_change(basis_change_q, basis_change_t, data.little[i].translation.x, data.little[i].translation.y, data.little[i].translation.z,
        data.little[i].rotation.x, data.little[i].rotation.y, data.little[i].rotation.z, data.little[i].rotation.w)
   
    
    if len(one_sample) == 10:
        return df
    df.loc[0] = one_sample
    return df

def visualize_in_rviz(df, left_flag):
    if df.shape[0] == 0:
        return

    global marker_size
    global markerArray
    global gelenke
    global rviz_pub
    global left_gelenke
    global right_gelenke
    hand_gelenke = 0
    if left_flag:
        hand_gelenke = left_gelenke
    else:
        hand_gelenke = right_gelenke

    for gelenk in hand_gelenke:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.get_rostime()
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = marker_size
        marker.scale.y = marker_size
        marker.scale.z = marker_size
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = df.loc[0,gelenk[0]]
        marker.pose.position.y = df.loc[0,gelenk[1]]
        marker.pose.position.z = 10
        marker.pose.orientation.x = df.loc[0,gelenk[3]]
        marker.pose.orientation.y = df.loc[0,gelenk[4]]
        marker.pose.orientation.z = df.loc[0,gelenk[5]]
        marker.pose.orientation.w = df.loc[0,gelenk[6]]
        markerArray.markers.append(marker)


    id = 0
    for m in markerArray.markers:
        m.id = id
        id += 1
    
        # Publish the MarkerArray
    rviz_pub.publish(markerArray)
    # marker muss vor jedem neuen pushen gecleart werden
    label = 0
    for m in markerArray.markers:
        if m.type == marker.TEXT_VIEW_FACING:
            label = m
            markerArray.markers = []
            markerArray.markers.append(label)
    if label == 0:
        markerArray.markers = []


def z_filter(df):

    global gelenke
    global left_gelenke
    global right_gelenke
    z_trashhold = -0.20
    z_labels = []
    if 'L_is_valid' in df.columns:
        z_labels.append('L_palm_j0_t_z')
    if 'R_is_valid' in df.columns:
        z_labels.append('R_palm_j0_t_z')
        
    if df.shape[0] > 0:
        minimum = np.min([df.loc[0,t_z] for t_z in z_labels])
        if minimum >= z_trashhold:
            return True
        else: 
            return False
    return False

def z_marker(x,y,z):
    
    global marker_size
    global markerArray
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.get_rostime()
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = marker_size
    marker.scale.y = marker_size
    marker.scale.z = marker_size
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.x = 1
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 0
    markerArray.markers.append(marker)

def callback_L(data):
    global sample_counter_L
    global sliding_window
    if sample_counter_L >= predict_every_n_th_sample:
        global tf_listener
        secs = data.header.stamp.secs
        nsecs = data.header.stamp.nsecs
        trans = 0
        rot = 0
        try:
            tf_listener.waitForTransform('/map', '/iviz_win_vr/xr/head',rospy.Time(secs, nsecs), rospy.Duration(1.0))
            (trans,rot) = tf_listener.lookupTransform('/map' , '/iviz_win_vr/xr/head',rospy.Time(secs, nsecs))
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        df = process_raw_hand_data_to_df(data,trans, rot, left_flag=True)
        if z_filter(df):
            z_marker(data.palm.translation.x, data.palm.translation.y, data.palm.translation.z)
        visualize_in_rviz(df, left_flag = True)
        merge_one_sampe(df, left_flag=True)
        return
    else:
        sample_counter_L += 1
        return
    
def callback_R(data):
    global sample_counter_R
    global sliding_window
    if sample_counter_R >= predict_every_n_th_sample:
        global tf_listener
        secs = data.header.stamp.secs
        nsecs = data.header.stamp.nsecs
        trans = 0
        rot = 0
        try:
            tf_listener.waitForTransform('/map', '/iviz_win_vr/xr/head',rospy.Time(secs, nsecs), rospy.Duration(1.0))
            (trans,rot) = tf_listener.lookupTransform('/map' , '/iviz_win_vr/xr/head',rospy.Time(secs, nsecs))
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        df = process_raw_hand_data_to_df(data,trans, rot, left_flag=False)
        if z_filter(df):
            z_marker(data.palm.translation.x, data.palm.translation.y, data.palm.translation.z)
        visualize_in_rviz(df, left_flag = False)
        merge_one_sampe(df, left_flag=False)
        return
    else:
        sample_counter_R += 1
        return

def callback_tf(data):
    return

def listener():
    rospy.Subscriber('/iviz_win_vr/xr/right_hand', XRHandState, callback_R)
    rospy.Subscriber('/iviz_win_vr/xr/left_hand', XRHandState, callback_L)
    rospy.Subscriber('/tf', TFMessage, callback_tf)
    rospy.spin()

def load_model():
    global model
    global l2_regularistion_flag
    global l2_regularistion_value
    print('Loading model form ' + model_path + '...')
    if not model_path:
        print('Error: No model path specified')
        exit()



    if T.cuda.is_available():
        model = T.load(model_path)
    else:
        model = T.load(model_path, map_location=T.device('cpu'))
    model.eval()
    if T.cuda.is_available():
        model.cuda()
    print('Finisehd loading')

def visualize_prediction(prediction):
    label = 'empty'
    if prediction == 0:
        label = 'no gesture'
    if prediction == 1:
        label = 'clap'
    if prediction == 2:
        label = 'grab and pull'
    if prediction == 4:
        label = 'left swipe'
    if prediction == 3:
        label = 'right swipe'
    if prediction == 5:
        label = 'rope'


    global markerArray
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.get_rostime()
    marker.type = marker.TEXT_VIEW_FACING
    marker.text = label
    marker.action = marker.ADD
    marker.scale.x = 10
    marker.scale.y = 10
    marker.scale.z = 10
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.position.x = 1
    marker.pose.position.y = 0
    marker.pose.position.z = 1
    marker.pose.orientation.x = 1
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 0
    markerArray.markers.append(marker)
    print(label)

def make_prediction():
    global model
    global sliding_window

    if 'secs' in sliding_window.columns:
        sliding_window = sliding_window.drop(columns=['secs'])
    if 'nsecs' in sliding_window.columns:
        sliding_window = sliding_window.drop(columns=['nsecs'])
    left_, right = DataCreator.get_all_column_names_seperatly()
    left_ = left_[2:]
    right = right[2:]

    # df in richtige reihenfolge bringen
    sliding_window = sliding_window[left_ + right]

    gesture_length = sliding_window.shape[0]
    padding_length = sliding_window_size - gesture_length 
    column_number = sliding_window.shape[1]
    zeros = pd.DataFrame([[0] * column_number] * padding_length, columns=sliding_window.columns)
    snip = zeros.append( sliding_window, ignore_index=True)

    snip = snip.fillna(0)
    snip['L_is_valid'] = np.where(snip['L_is_valid'] == True,1 ,0)
    snip['R_is_valid'] = np.where(snip['R_is_valid'] == True,1 ,0)
    snip_np = np.array(snip)
    snip_tensor = T.Tensor(snip_np)

    if T.cuda.is_available():
        input = snip_tensor.view(-1, sliding_window_size,input_dim).cuda()
    else:
        input = snip_tensor.view(-1,sliding_window_size,input_dim)

    output = model(input)
    _ ,predicted = T.max(output.data, 1)
    prediction = np.array(predicted.cpu())[0]
    print('predicted gesture', prediction)
    visualize_prediction(prediction)
    return

def init_gelenke():
    global gelenke
    global left_gelenke
    global right_gelenke
    column = DataCreator.get_all_column_names()
    column_names = list(filter(lambda x: "_t_" in x or "_r_" in x, column))
    for i in range(0, len(column_names),7): 
        gelenk = []
        for j in range(7):
            gelenk.append(column_names[i + j])
        gelenke.append(gelenk)

    for gelenk in gelenke:
        gelenk = list(filter(lambda x: 'L' in x, gelenk))
        if len(gelenk) != 0:
            left_gelenke.append(gelenk)

    for gelenk in gelenke:
        gelenk = list(filter(lambda x: 'R' in x, gelenk))
        if len(gelenk) != 0:
            right_gelenke.append(gelenk)

def init_sliding_windows():
    global sliding_window
    global all_columns
    sliding_window = pd.DataFrame(data=None, index=None, dtype=None, copy=False, columns=all_columns)




init_sliding_windows()
init_gelenke()


def adjust_sliding_window_size(sliding_window):
    global sliding_window_size
    while sliding_window.shape[0] > sliding_window_size:
        sliding_window = sliding_window.iloc[:-1 , :]
    return sliding_window


def merge_one_sampe(df, left_flag):
    if df.shape[0] == 0:
        return
    global one_sample
    global sliding_window
    sliding_window = sliding_window.loc[:,~sliding_window.columns.duplicated()]
    one_sample = one_sample.loc[:,~one_sample.columns.duplicated()]
    one_sample['secs'] = 1
    df = df.drop(columns=['nsecs'])
    df['secs'] = 1
    # wenn es leer ist
    if one_sample.shape[0] == 0:
        one_sample = pd.merge(one_sample, df, on=['secs'], how='outer')

    elif left_flag and 'L_is_valid' in one_sample.columns:
        # one sample zum window hinzufuegen
        # wenn z_filter true dann zum sliding window hinzufuegen
        if z_filter(one_sample):
            sliding_window = pd.concat([one_sample, sliding_window])
            sliding_window = adjust_sliding_window_size(sliding_window)
        # wenn t_filter false dann nicht hinzufuegen, sondern prediction machen und window clearen
        else:
            if sliding_window.shape[0] != 0:
                make_prediction()
                sliding_window = pd.DataFrame(data=None, index=None, dtype=None, copy=False, columns=all_columns)
        # neues one sample machen
        one_sample = pd.DataFrame(data=None, index=None, dtype=None, copy=False, columns=['secs'])
        one_sample['secs'] = 1
        one_sample = pd.merge(one_sample, df, on=['secs'], how='outer')
    elif left_flag and 'L_is_valid' not in one_sample.columns:
        one_sample = pd.merge(one_sample, df, on=['secs'], how='outer')
    elif not left_flag and 'R_is_valid' in one_sample.columns:
        # one sample zum window hinzufuegen
        if z_filter(one_sample):
            sliding_window = pd.concat([one_sample, sliding_window])
            sliding_window = adjust_sliding_window_size(sliding_window)
        else:
            if sliding_window.shape[0] != 0:
                make_prediction()
                sliding_window = pd.DataFrame(data=None, index=None, dtype=None, copy=False, columns=all_columns)
        # neues one sample machen
        one_sample = pd.DataFrame(data=None, index=None, dtype=None, copy=False, columns=['secs'])
        one_sample['secs'] = 1
        one_sample = pd.merge(one_sample, df, on=['secs'], how='outer')
    elif not left_flag and 'R_is_valid' not in one_sample.columns:
        one_sample = pd.merge(one_sample, df, on=['secs'], how='outer')
load_model()
listener()





