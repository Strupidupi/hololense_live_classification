import sys
sys.path.append('/home/telepresence/catkin_ws/src/iviz_msgs')
import methods
#from data_creator.methods import get_data_from_bag_file, add_labels_right_hand, add_labels_both_hands, get_column_names, update_overall_csv, convert_bag_to_csv


class DataCreator:
    def __init__(self):
        self.data = None

    @staticmethod
    def get_df_from_bag_file(path):
        return methods.get_data_from_bag_file(path)

    @staticmethod
    def get_all_column_names():
        l, r = methods.get_column_names()
        return l + r

    @staticmethod
    def get_all_column_names_seperatly():
        return methods.get_column_names()

    @staticmethod
    def save_bag_data_to_csv(path_bag_file, path_csv_file):
        df = methods.get_data_from_bag_file(path_bag_file)
        df.to_csv(path_csv_file)

    @staticmethod
    def save_all_bag_data_to_csv():
        methods.convert_bag_to_csv(bag_path='data/bag', csv_path='data/csv')

    @staticmethod
    def add_labels_right_hand_to_df(df, gesture_number):
        return methods.add_labels_right_hand(df, gesture_number)

    @staticmethod
    def add_labels_both_hands_to_df(df, gesture_number):
        return methods.add_labels_both_hands(df, gesture_number)

    @staticmethod
    def update_csv_with_all_gestures(file_name = 'all_gestures.csv'):
        methods.update_overall_csv(csv_path = "data/csv/" + file_name, bag_path = 'data/bag')
