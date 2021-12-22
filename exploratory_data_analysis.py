import logging
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# noinspection PyUnresolvedReferences
class DataAnalyser:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/exploratory_data_analyser.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.train = pd.read_csv('.csv files/train.csv')
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.buf1)
        self.convert_age_to_years()
        self.age_count_plot()
        self.create_train_dog_cat()
        self.create_age_plot()
        self.create_sex_outcome_plot()
        self.create_gender_plot()
        self.fill_na_name()
        self.create_name_plot()
        self.create_type_plot()
        self.logger.debug('Closing Class')

    def df_current_state(self, buf):
        self.logger.debug(f"Current train.head()\n{self.train.head()}")
        self.train.info(buf=buf)
        self.logger.debug(f"Current train.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current train.describe()\n{self.train.describe(include='all')}")

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

    def age_count_plot(self):
        self.logger.debug(f'Generating age_count_plot.png')
        fig, ax = plt.subplots()
        sns.displot(data=self.train, x='AgeuponOutcome', hue='AnimalType', bins=20, multiple="stack", ax=ax)
        plt.xticks(range(20))
        plt.savefig('plots/age_count_plot.png')

    def create_train_dog_cat(self):
        self.train_cat = self.train.loc[self.train['AnimalType'] == 'Cat']
        self.train_dog = self.train.loc[self.train['AnimalType'] == 'Dog']

    def plot_frac(self, df, x_label, ganimal, gtype, name):
        # Creating aliases to reduce the length of code
        ds1d, ds2d, ds3d, ds4d, ds5d = df['Adoption'], df['Died'], df['Euthanasia'], df['Return_to_owner'], df[
            'Transfer']

        fig, ax = plt.subplots()
        # Creating each of the bars, passing the bottom parameter as the sum of the bars under it
        ax.bar(df.index, ds1d, label='Adoption')
        ax.bar(df.index, ds2d, label='Died', bottom=ds1d)
        ax.bar(df.index, ds3d, label='Euthanasia', bottom=np.array(ds1d) + np.array(ds2d))
        ax.bar(df.index, ds4d, label='Return_to_owner', bottom=np.array(ds1d) + np.array(ds2d) + np.array(ds3d))
        ax.bar(df.index, ds5d, label='Transfer',
               bottom=np.array(ds1d) + np.array(ds2d) + np.array(ds3d) + np.array(ds4d))
        ax.legend()
        # Set the x-axis to the animal gender
        ax.set_xticklabels(x_label)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(range(50)))
        plt.xticks(rotation=30)
        plt.yticks(np.linspace(0, 1, 11))
        plt.title('Outcome of ' + ganimal + ' grouped by ' + gtype)
        plt.ylabel('Fraction of outcomes')
        plt.savefig(f'plots/{name}_plot.png')

    def create_age_plot1(self):
        # Creates the dataframe that will be filled with values to be plotted
        self.df_plot1 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        # Defines range used in the for loop
        max_year = int(self.train_cat['AgeuponOutcome'].max())
        # Loop that fills the df_plot1
        for i in range(max_year):
            # Selects the values of train_cat for the year i
            train_cat_year = self.train_cat[(self.train_cat['AgeuponOutcome'] >= i) & (self.train_cat['AgeuponOutcome'] < i + 1)]
            # Groups number of outcomes in current year and normalizes it
            val_year = train_cat_year.groupby(['OutcomeType']).count()['AgeuponOutcome'] / train_cat_year[
                'AgeuponOutcome'].count()
            # Append values of current year in the dataframe
            self.df_plot1 = self.df_plot1.append(val_year)
        # Sets the dataframe index from 0 to max year count - 1
        self.df_plot1.set_index(np.array(range(max_year)), inplace=True)
        self.df_plot1.fillna(0, inplace=True)

    def create_age_plot2(self):
        # Creates the dataframe that will be filled with values to be plotted
        self.df_plot2 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])

        # Defines range used in the for loop
        max_year = int(self.train_dog['AgeuponOutcome'].max())
        # Loop that fills the df_plot1
        for i in range(max_year):
            # Selects the values of train_cat for the year i
            train_dog_year = self.train_dog[(self.train_dog['AgeuponOutcome'] >= i) & (self.train_dog['AgeuponOutcome'] < i + 1)]
            # Groups number of outcomes in current year and normalizes it
            val_year = train_dog_year.groupby(['OutcomeType']).count()['AgeuponOutcome'] / train_dog_year[
                'AgeuponOutcome'].count()
            # Append values of current year in the dataframe
            self.df_plot2 = self.df_plot2.append(val_year)
        # Sets the dataframe index from 0 to max year count - 1
        self.df_plot2.set_index(np.array(range(max_year)), inplace=True)
        self.df_plot2.fillna(0, inplace=True)

    def create_age_plot(self):
        self.create_age_plot1()
        self.create_age_plot2()
        self.plot_frac(self.df_plot1, range(20), 'Cats', 'Age', 'cat_age')
        self.plot_frac(self.df_plot2, range(20), 'Dogs', 'Age', 'dog_age')

    def create_sex_outcome_plot(self):
        self.logger.debug(f'Generating sex_outcome_plot.png')
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.countplot(data=self.train, x='SexuponOutcome', ax=ax)
        plt.xticks(rotation=30)
        plt.savefig(f'plots/sex_outcome_plot.png')

    def create_gender_plot1(self):
        self.df_plot1 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        # Defines range used in the for loop
        self.total_sex = ['Neutered Male', 'Spayed Female', 'Intact Male', 'Intact Female', 'Unknown']
        # Loop that fills the df_plot1
        for i in self.total_sex:
            # Selects the values of train_cat for the sex i
            train_cat_sex = self.train_cat[self.train_cat['SexuponOutcome'] == i]
            # Groups number of outcomes in current sex and normalizes it
            val_sex = train_cat_sex.groupby(['OutcomeType']).count()['SexuponOutcome'] / train_cat_sex[
                'SexuponOutcome'].count()
            # Append values of current sex in the dataframe
            self.df_plot1 = self.df_plot1.append(val_sex)
        # Sets the dataframe index from 0 to number of sexes - 1
        self.df_plot1.set_index(np.array(range(len(self.total_sex))), inplace=True)
        self.df_plot1.fillna(0, inplace=True)

    def create_gender_plot2(self):
        self.df_plot2 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        # Loop that fills the df_plot1
        for i in self.total_sex:
            # Selects the values of train_cat for the sex i
            train_dog_sex = self.train_dog[self.train_dog['SexuponOutcome'] == i]
            # Groups number of outcomes in current sex and normalizes it
            val_sex = train_dog_sex.groupby(['OutcomeType']).count()['SexuponOutcome'] / train_dog_sex[
                'SexuponOutcome'].count()
            # Append values of current sex in the dataframe
            self.df_plot2 = self.df_plot2.append(val_sex)
        # Sets the dataframe index from 0 to number of sexes - 1
        self.df_plot2.set_index(np.array(range(len(self.total_sex))), inplace=True)
        self.df_plot2.fillna(0, inplace=True)

    def create_gender_plot(self):
        self.create_gender_plot1()
        self.create_gender_plot2()
        self.plot_frac(self.df_plot1, self.total_sex, 'Cats', 'Gender', 'cat_gender')
        self.plot_frac(self.df_plot2, self.total_sex, 'Dogs', 'Gender', 'dog_gender')

    def fill_na_name(self):
        self.train_cat, self.train_dog = self.train_cat.fillna('DummyName'), self.train_dog.fillna('DummyName')

    def create_name_plot1(self):
        self.total_name = ['Named', 'Unnamed']
        # Creates the dataframe that will be filled with values to be plotted
        self.df_plot1 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        train_cat_name = self.train_cat[self.train_cat['Name'] != 'DummyName']
        # Groups number of outcomes in current name and normalizes it
        val_name = train_cat_name.groupby(['OutcomeType']).count()['Name'] / train_cat_name['Name'].count()
        # Append values of current name in the dataframe
        self.df_plot1 = self.df_plot1.append(val_name)
        train_cat_name = self.train_cat[self.train_cat['Name'] == 'DummyName']
        # Groups number of outcomes in current name and normalizes it
        val_name = train_cat_name.groupby(['OutcomeType']).count()['Name'] / train_cat_name['Name'].count()
        # Append values of current name in the dataframe
        self.df_plot1 = self.df_plot1.append(val_name)
        # Sets the dataframe index from 0 to number of names - 1
        self.df_plot1.set_index(np.array(range(2)), inplace=True)
        self.df_plot1.fillna(0, inplace=True)

    def create_name_plot2(self):
        self.df_plot2 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        train_dog_name = self.train_dog[self.train_dog['Name'] != 'DummyName']
        # Groups number of outcomes in current name and normalizes it
        val_name = train_dog_name.groupby(['OutcomeType']).count()['Name'] / train_dog_name['Name'].count()
        # Append values of current name in the dataframe
        self.df_plot2 = self.df_plot2.append(val_name)
        train_dog_name = self.train_dog[self.train_dog['Name'] == 'DummyName']
        # Groups number of outcomes in current name and normalizes it
        val_name = train_dog_name.groupby(['OutcomeType']).count()['Name'] / train_dog_name['Name'].count()
        # Append values of current name in the dataframe
        self.df_plot2 = self.df_plot2.append(val_name)
        # Sets the dataframe index from 0 to number of names - 1
        self.df_plot2.set_index(np.array(range(2)), inplace=True)
        self.df_plot2.fillna(0, inplace=True)

    def create_name_plot(self):
        self.create_name_plot1()
        self.create_name_plot2()
        self.plot_frac(self.df_plot1, self.total_name, 'Cats', 'Name', 'cat_name')
        self.plot_frac(self.df_plot2, self.total_name, 'Dogs', 'Name', 'dog_name')

    def create_type_plot1(self):
        self.total_type = ['Cat', 'Dog']
        # Creates the dataframe that will be filled with values to be plotted
        self.df_plot1 = pd.DataFrame(columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        train_type = self.train[self.train['AnimalType'] != 'Dog']
        # Groups number of outcomes in current name and normalizes it
        val_type = train_type.groupby(['OutcomeType']).count()['AnimalType'] / train_type['AnimalType'].count()
        # Append values of current name in the dataframe
        self.df_plot1 = self.df_plot1.append(val_type)
        train_type = self.train[self.train['AnimalType'] == 'Dog']
        # Groups number of outcomes in current name and normalizes it
        val_type = train_type.groupby(['OutcomeType']).count()['AnimalType'] / train_type['AnimalType'].count()
        # Append values of current name in the dataframe
        self.df_plot1 = self.df_plot1.append(val_type)
        # Sets the dataframe index from 0 to number of names - 1
        self.df_plot1.set_index(np.array(range(2)), inplace=True)
        self.df_plot1.fillna(0, inplace=True)

    def create_type_plot(self):
        self.create_type_plot1()
        self.plot_frac(self.df_plot1, self.total_type, 'Animals', 'Animal Type', 'animal_type')
