import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion start')
        try:
            # Read data from CSV file into a DataFrame
            df = pd.read_csv(self.csv_file_path)

            # Define the new column names
            new_columns = {
                'class': 'class',
                'cap-shape': 'cap_shape',
                'cap-surface': 'cap_surface',
                'cap-color': 'cap_color',
                'bruises': 'bruises',
                'odor': 'odor',
                'gill-attachment': 'gill_attachment',
                'gill-spacing': 'gill_spacing',
                'gill-size': 'gill_size',
                'gill-color': 'gill_color',
                'stalk-shape': 'stalk_shape',
                'stalk-root': 'stalk_root',
                'stalk-surface-above-ring': 'stalk_surface_above_ring',
                'stalk-surface-below-ring': 'stalk_surface_below_ring',
                'stalk-color-above-ring': 'stalk_color_above_ring',
                'stalk-color-below-ring': 'stalk_color_below_ring',
                'veil-type': 'veil_type',
                'veil-color': 'veil_color',
                'ring-number': 'ring_number',
                'ring-type': 'ring_type',
                'spore-print-color': 'spore_print_color',
                'population': 'population',
                'habitat': 'habitat'
            }
            df = df.rename(columns=new_columns)
            logging.info('Dataset read data as pandas dataframe')
            logging.info(f'{df.head()}')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('raw data save')

            df.drop(columns=['veil_type'], axis=1, inplace=True)
            logging.info("Dropping veil type from dataframe")

            # Check for '?' in each column
            columns_with_question_mark = df.columns[df.apply(lambda col: col.str.contains('\?', na=False)).any()]
            print("columns_with_question_mark name: ", columns_with_question_mark)

            # Replace '?' with None
            df.replace('?', None, inplace=True)
            df.fillna(df.mode().iloc[0], inplace=True)
            logging.info("Removing ? and replace with mode value")

            logging.info('train test split')
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=101)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occurred at data ingestion stage')
            raise CustomException(e, sys)

if __name__ == '__main__':
    # CSV File Configuration
    csv_file_path = "notebooks/data/mushrooms.csv"

    # Initialize DataIngestion with CSV file path
    obj = DataIngestion(csv_file_path)

    # Run Data Ingestion
    obj.initiate_data_ingestion()
