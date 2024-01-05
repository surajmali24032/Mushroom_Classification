import os
import sys
import numpy as np
import pandas as pd


from category_encoders.cat_boost import CatBoostEncoder
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info('Read train and test data')

            class_map = {'e':0, 'p':1}

            train_df['class'] = train_df['class'].map(class_map)
            test_df['class'] = test_df['class'].map(class_map)
            logging.info('Target column encoded')

            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            # Define catboost encoder
            cat_encoder = CatBoostEncoder()

            target_column_name = 'class'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming using preprocessor obj
            input_feature_train_arr = cat_encoder.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = cat_encoder.transform(input_feature_test_df)
            logging.info("Transforming categorical features into Numerical features")

            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=cat_encoder
            )
            logging.info("Data transformation Successfully")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)            

