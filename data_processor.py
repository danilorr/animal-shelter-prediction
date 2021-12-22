import logging
from io import StringIO
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



class DataProcessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/data_processor.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.train = pd.read_csv('.csv files/train.csv', infer_datetime_format=True, parse_dates=[2])
        self.test = pd.read_csv('.csv files/test.csv', infer_datetime_format=True, parse_dates=[2])
        self.buf1 = StringIO()
        self.buf2 = StringIO()
        self.buf3 = StringIO()
        self.buf4 = StringIO()
        self.df_list = [self.train, self.test]

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.buf1, self.train, 'train')
        self.df_current_state(self.buf2, self.test, 'test')
        self.convert_age_to_years()
        for df in self.df_list:
            self.create_is_intact_feature(df)
            self.convert_sex_to_isfemale(df)
            self.convert_name_to_hasname(df)
            self.convert_animaltype_to_iscat(df)
        self.drop_low_correlation_features()
        self.df_current_state(self.buf3, self.train, 'train')
        self.df_current_state(self.buf4, self.test, 'test')
        self.create_train_df_csv()
        self.create_test_df_csv()
        self.logger.debug('Closing Class')

    def df_current_state(self, buf, df, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current {name}.describe()\n{df.describe(datetime_is_numeric=True, include='all')}")

    def age_to_years(self, item):
        if type(item) is str:
            item = [item]
        ages_in_years = np.zeros(len(item))
        for i in range(len(item)):
            if type(item.get(i)) is str:
                if 'day' in item[i]:
                    ages_in_years[i] = int(item[i].split(' ')[0]) / 365
                if 'week' in item[i]:
                    ages_in_years[i] = int(item[i].split(' ')[0]) / 52.1429
                if 'month' in item[i]:
                    ages_in_years[i] = int(item[i].split(' ')[0]) / 12
                if 'year' in item[i]:
                    ages_in_years[i] = int(item[i].split(' ')[0])
            else:
                ages_in_years[i] = 0
        return ages_in_years

    def convert_age_to_years(self):
        self.train['AgeuponOutcome'] = self.age_to_years(self.train['AgeuponOutcome']).round(1)
        self.test['AgeuponOutcome'] = self.age_to_years(self.test['AgeuponOutcome']).round(1)

    def create_is_intact_feature(self, df):
        df['SexuponOutcome'].fillna('Unknown', inplace=True)
        df['IsIntact'] = df.loc[df['SexuponOutcome'] == 'Unknown', 'SexuponOutcome']
        df.loc[~df['IsIntact'].isnull(), 'IsIntact'] = -1
        df.loc[df['SexuponOutcome'].str.contains('Intact'), 'IsIntact'] = 1
        df['IsIntact'].fillna(0, inplace=True)
        df['IsIntact'] = pd.to_numeric(df['IsIntact'])

    def convert_sex_to_isfemale(self, df):
        df.rename(columns={'SexuponOutcome': 'IsFemale'}, inplace=True)
        df.loc[df['IsFemale'].str.contains('Female'), 'IsFemale'] = '1'
        df.loc[df['IsFemale'].str.contains('Male'), 'IsFemale'] = 0
        df.loc[df['IsFemale'] == 'Unknown', 'IsFemale'] = -1
        df['IsFemale'] = pd.to_numeric(df['IsFemale'])

    def convert_name_to_hasname(self, df):
        df.rename(columns={'Name': 'HasName'}, inplace=True)
        df['HasName'].fillna(0, inplace=True)
        df.loc[df['HasName'] != 0, 'HasName'] = 1
        df['HasName'] = pd.to_numeric(df['HasName'])

    def convert_animaltype_to_iscat(self, df):
        df.rename(columns={'AnimalType': 'IsCat'}, inplace=True)
        df.loc[df['IsCat'] == 'Cat', 'IsCat'] = 1
        df.loc[df['IsCat'] != 1, 'IsCat'] = 0
        df['IsCat'] = pd.to_numeric(df['IsCat'])

    def drop_low_correlation_features(self):
        self.train.drop(['AnimalID', 'DateTime', 'OutcomeSubtype', 'Breed', 'Color'], axis=1, inplace=True)
        self.test.drop(['ID', 'DateTime', 'Breed', 'Color'], axis=1, inplace=True)

    def create_train_df_csv(self):
        self.logger.debug('Creating train.csv')
        self.train.to_csv(r'.csv files/train_processed.csv')

    def create_test_df_csv(self):
        self.logger.debug('Creating test.csv')
        self.test.to_csv(r'.csv files/test_processed.csv')
