import os
import shutil
from os.path import exists

import pandas as pd
import rosbag
import quaternion
from pandas import DataFrame, read_csv
import math
import numpy as np


def get_data_from_bag_file(path_bag_file):
    def rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w, ):
        pt = np.quaternion(0, t_x, t_y, t_z, )
        pq = np.quaternion(r_w, r_x, r_y, r_z)

        pt = basis_change["q"] * pt * basis_change["q"].conjugate() + basis_change["t"]
        pq = basis_change["q"] * pq

        return pt.x, pt.y, pt.z, pq.x, pq.y, pq.z, pq.w

    bag = rosbag.Bag(path_bag_file)

    columns_left_hand, columns_right_hand = get_column_names()

    data = {'left_hand': DataFrame(data=None, index=None, columns=columns_left_hand, dtype=None, copy=False),
            'right_hand': DataFrame(data=None, index=None, columns=columns_right_hand, dtype=None, copy=False)}

    counter = 0
    basis_change = {}

    # build dataframy by appending rows
    i = 0
    for topic, msg, t in bag.read_messages():
        # if i < 7:
        #     i += 1
        #     continue
        if topic == "/tf":
            t = np.quaternion(0, msg.transforms[-1].transform.translation.x, msg.transforms[-1].transform.translation.y,
                              msg.transforms[-1].transform.translation.z)
            q = np.quaternion(msg.transforms[-1].transform.rotation.w, msg.transforms[-1].transform.rotation.x,
                              msg.transforms[-1].transform.rotation.y, msg.transforms[-1].transform.rotation.z)

            q_i = q.inverse()

            basis_change["t"] = q_i * -t * q_i.conjugate()
            basis_change["q"] = q_i

        for key in data.keys():
            if ('iviz_win_vr/xr/'+key) in topic:
                row = []
                row.extend([msg.header.stamp.secs, msg.header.stamp.nsecs, msg.is_valid])

                if msg.palm:
                    t_x, t_y, t_z = msg.palm.translation.x, msg.palm.translation.y, msg.palm.translation.z
                    r_x, r_y, r_z, r_w = msg.palm.rotation.x, msg.palm.rotation.y, msg.palm.rotation.z, \
                                         msg.palm.rotation.w

                    t_x, t_y, t_z, r_x, r_y, r_z, r_w = \
                        rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w)

                    row.extend([t_x, t_y, t_z, r_x, r_y, r_z, r_w])
                else:
                    for i in range(7):
                        row.append(math.nan)

                if msg.thumb:
                    for i in range(len(msg.thumb)):
                        t_x, t_y, t_z = msg.thumb[i].translation.x, msg.thumb[i].translation.y, msg.thumb[
                            i].translation.z
                        r_x, r_y, r_z, r_w = msg.thumb[i].rotation.x, msg.thumb[i].rotation.y, msg.thumb[
                            i].rotation.z, \
                                             msg.thumb[i].rotation.w
                        t_x, t_y, t_z, r_x, r_y, r_z, r_w = \
                            rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w)

                        row.extend([t_x, t_y, t_z, r_x, r_y, r_z, r_w])
                else:
                    for i in range(35):
                        row.append(math.nan)

                if msg.index:
                    for i in range(len(msg.index)):
                        t_x, t_y, t_z = msg.index[i].translation.x, msg.index[i].translation.y, msg.index[
                            i].translation.z
                        r_x, r_y, r_z, r_w = msg.index[i].rotation.x, msg.index[i].rotation.y, msg.index[
                            i].rotation.z, \
                                             msg.index[i].rotation.w
                        t_x, t_y, t_z, r_x, r_y, r_z, r_w = \
                            rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w)
                        row.extend([t_x, t_y, t_z, r_x, r_y, r_z, r_w])
                else:
                    for i in range(35):
                        row.append(math.nan)

                if msg.middle:
                    for i in range(len(msg.middle)):
                        t_x, t_y, t_z = msg.middle[i].translation.x, msg.middle[i].translation.y, msg.middle[
                            i].translation.z
                        r_x, r_y, r_z, r_w = msg.middle[i].rotation.x, msg.middle[i].rotation.y, msg.middle[
                            i].rotation.z, \
                                             msg.middle[i].rotation.w
                        t_x, t_y, t_z, r_x, r_y, r_z, r_w = \
                            rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w)
                        row.extend([t_x, t_y, t_z, r_x, r_y, r_z, r_w])
                else:
                    for i in range(35):
                        row.append(math.nan)

                if msg.ring:
                    for i in range(len(msg.ring)):
                        t_x, t_y, t_z = msg.ring[i].translation.x, msg.ring[i].translation.y, msg.ring[
                            i].translation.z
                        r_x, r_y, r_z, r_w = msg.ring[i].rotation.x, msg.ring[i].rotation.y, msg.ring[i].rotation.z, \
                                             msg.ring[i].rotation.w
                        t_x, t_y, t_z, r_x, r_y, r_z, r_w = \
                            rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w)
                        row.extend([t_x, t_y, t_z, r_x, r_y, r_z, r_w])
                else:
                    for i in range(35):
                        row.append(math.nan)

                if msg.little:
                    for i in range(len(msg.little)):
                        t_x, t_y, t_z = msg.little[i].translation.x, msg.little[i].translation.y, msg.little[
                            i].translation.z
                        r_x, r_y, r_z, r_w = msg.little[i].rotation.x, msg.little[i].rotation.y, msg.little[
                            i].rotation.z, \
                                             msg.little[i].rotation.w
                        t_x, t_y, t_z, r_x, r_y, r_z, r_w = \
                            rotate_to_head(t_x, t_y, t_z, r_x, r_y, r_z, r_w)
                        row.extend([t_x, t_y, t_z, r_x, r_y, r_z, r_w])
                else:
                    for i in range(35):
                        row.append(math.nan)

                data[key].loc[counter] = row
        counter += 1
    bag.close()

    return pd.merge(data['left_hand'], data['right_hand'], on=["secs", "nsecs"], how="outer")


def get_column_names():
    # segments: 'palm', 'thumb', 'index', 'middle', 'ring', 'little'
    # build columns
    columns = ['secs', 'nsecs', 'is_valid', 'palm_j0_t_x', 'palm_j0_t_y', 'palm_j0_t_z', 'palm_j0_r_x',
               'palm_j0_r_y', 'palm_j0_r_z', 'palm_j0_r_w']
    n = 5  # number of joints

    for i in range(n):
        columns.extend(['thumb_j'+str(i)+'_t_x', 'thumb_j'+str(i)+'_t_y', 'thumb_j'+str(i)+'_t_z', 'thumb_j'+str(i)+'_r_x',
                        'thumb_j'+str(i)+'_r_y', 'thumb_j'+str(i)+'_r_z', 'thumb_j'+str(i)+'_r_w'])
    for i in range(n):
        columns.extend(['index_j'+str(i)+'_t_x', 'index_j'+str(i)+'_t_y', 'index_j'+str(i)+'_t_z', 'index_j'+str(i)+'_r_x',
                        'index_j'+str(i)+'_r_y', 'index_j'+str(i)+'_r_z', 'index_j'+str(i)+'_r_w'])
    for i in range(n):
        columns.extend(['middle_j'+str(i)+'_t_x', 'middle_j'+str(i)+'_t_y', 'middle_j'+str(i)+'_t_z', 'middle_j'+str(i)+'_r_x',
                        'middle_j'+str(i)+'_r_y', 'middle_j'+str(i)+'_r_z', 'middle_j'+str(i)+'_r_w'])
    for i in range(n):
        columns.extend(['ring_j'+str(i)+'_t_x', 'ring_j'+str(i)+'_t_y', 'ring_j'+str(i)+'_t_z', 'ring_j'+str(i)+'_r_x',
                        'ring_j'+str(i)+'_r_y', 'ring_j'+str(i)+'_r_z', 'ring_j'+str(i)+'_r_w'])
    for i in range(n):
        columns.extend(['little_j'+str(i)+'_t_x', 'little_j'+str(i)+'_t_y', 'little_j'+str(i)+'_t_z', 'little_j'+str(i)+'_r_x',
                        'little_j'+str(i)+'_r_y', 'little_j'+str(i)+'_r_z', 'little_j'+str(i)+'_r_w'])

    # rename the columns, so that left and right can be distinguished example: L_thumb_j0_t_x and R_thumb_j0_t_x
    c_left_hand = list(map(lambda x: "L_" + x if (not (x == "secs" or x == "nsecs")) else x, columns))
    c_right_hand = list(map(lambda x: "R_" + x if (not (x == "secs" or x == "nsecs")) else x, columns))

    return c_left_hand, c_right_hand


def add_labels_right_hand(df, gesture_number):
    df['start'] = 0
    df['end'] = np.where(df['R_is_valid'] == False, 1, 0)
    df.loc[0, 'end'] = 0
    df['label'] = np.where(df['R_is_valid'] == True, gesture_number, 0)

    if df['R_is_valid'][0] == True:
        df.loc[0, 'start'] = 1

    for index in range(df.shape[0] - 1):
        cond = (df['R_is_valid'][index] != True and df['R_is_valid'][index + 1] == True)

        if cond:
            df.loc[index + 1, 'start'] = 1

    return df


def add_labels_both_hands(df, gesture_number):
    df['start'], df['end'], df['label'] = 0, 0, 0

    if df['R_is_valid'][0] == True and df['L_is_valid'][0] == True:
        df.loc[0, 'start'] = 1

    for index in range(df.shape[0] - 1):
        if df['R_is_valid'][index] == True and df['L_is_valid'][index] == True:
            df.loc[index, 'label'] = gesture_number

        if index == 0:
            continue

        cond1 = (df['R_is_valid'][index] == True and df['L_is_valid'][index] == True
                 and not (df['R_is_valid'][index - 1] == True and df['L_is_valid'][index - 1] == True))

        if cond1:
            df.loc[index, 'start'] = 1

        cond2 = df['R_is_valid'][index] == False or df['L_is_valid'][index] == False

        if cond2:
            df.loc[index, 'end'] = 1

    return df


def add_zero_labels(df):
    df['start'], df['end'], df['label'] = 0, 0, 0

    return df


def convert_bag_to_csv(bag_path, csv_path):
    directories = []

    for subdir, dirs, files in os.walk(bag_path):
        directories.extend(dirs)
        break

    for directory in directories:
        for file_name in os.listdir(bag_path +'/'+ directory):
            print('Converting' + file_name + '...')
            if directory == 'processed_bag_files':
                continue

            df = get_data_from_bag_file(bag_path +'/'+ directory +'/'+ file_name)

            df.drop(columns=['secs', 'nsecs'], axis=1, inplace=True)
            df['R_is_valid'] = np.where(df['R_is_valid'] == True, 1, 0)
            df['L_is_valid'] = np.where(df['L_is_valid'] == True, 1, 0)
            df.to_csv(csv_path  +'/'+ file_name.replace(".bag",".csv"), index=False)

            shutil.move(bag_path +'/'+ directory+ '/'+file_name,
                        bag_path +'/processed_bag_files/'+ directory  +'/'+ file_name)


def update_overall_csv(csv_path, bag_path):
    file_exists = exists(csv_path)

    dfs = []

    if file_exists:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    directories = []

    for subdir, dirs, files in os.walk(bag_path):
        directories.extend(dirs)
        break

    for directory in directories:
        for file_name in os.listdir(bag_path +'/'+ directory):
            if directory == 'processed_bag_files':
                continue

            df = get_data_from_bag_file(bag_path +'/'+ directory +'/'+ file_name)

            label = int(directory[0])

            if label == 0:
                frame = add_zero_labels(df)
            elif label == 1 or label == 5:  # two hands
                frame = add_labels_both_hands(df, label)
            else:
                frame = add_labels_right_hand(df, label)

            dfs.append(frame)

            shutil.move(bag_path +'/'+ directory +'/'+ file_name,
                        bag_path + '/processed_bag_files/' + directory +'/'+ file_name)

    df = pd.concat(dfs, ignore_index=True)

    df.drop(columns=['secs', 'nsecs'], axis=1, inplace=True)
    df['R_is_valid'] = np.where(df['R_is_valid'] == True, 1, 0)
    df['L_is_valid'] = np.where(df['L_is_valid'] == True, 1, 0)

    df.to_csv(csv_path, index=False)
