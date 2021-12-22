import logging
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/final_model_trainer.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.train = pd.read_csv('.csv files/train_processed.csv', index_col=0)
        self.test = pd.read_csv('.csv files/test_processed.csv', index_col=0)
        self.forest = self.forest = RandomForestClassifier(random_state=42)
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.make_test_probability_like()
        self.train_model()
        self.make_prediction()
        self.create_prediction_csv()
        self.create_prediction_heatmap()
        self.start_shap()
        self.summary_plots()
        self.age_dependency_plot()
        self.logger.debug('Closing Class')

    def df_current_state(self, df, buf, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current {name}.describe()\n{df.describe(include='all')}")

    def create_x_and_y_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y = self.train['OutcomeType']
        self.X = self.train.drop(['OutcomeType'], axis=1)

    def make_train_test_split(self):
        self.logger.debug('Creating train test split')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                stratify=self.y, random_state=42)

    def make_test_probability_like(self):
        self.y_test_conf = self.y_test.to_numpy()
        self.y_test = pd.get_dummies(self.y_test)
        self.y_test = self.y_test.to_numpy().astype(np.float32)

    def train_model(self):
        self.logger.debug('Training model')
        self.forest = RandomForestClassifier(criterion='entropy', max_leaf_nodes=83, min_samples_split=10,
                                             n_estimators=173, random_state=42)
        self.forest.fit(self.X, self.y)

    def make_prediction(self):
        self.logger.debug('Predicting test df')
        self.y_pred = self.forest.predict_proba(self.test)

    def create_prediction_csv(self):
        prediction = pd.DataFrame(data=self.y_pred,
                                  columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
        prediction['ID'] = (prediction.index + 1)
        prediction = prediction[['ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']]
        prediction = prediction.round().astype(int)
        prediction.to_csv('.csv files/test_prediction.csv', index=False)

    def create_prediction_heatmap(self):
        self.logger.debug(f'Generating prediction_heatmap.png')
        y_pred_conf = self.forest.predict(self.X_test)
        labels_cm = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
        cm = confusion_matrix(self.y_test_conf, y_pred_conf)
        df_cm = pd.DataFrame(cm, index=[i for i in labels_cm], columns=[i for i in labels_cm])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='YlGnBu', ax=ax)
        plt.savefig('plots/prediction_heatmap.png')
        plt.figure().clear()

    def start_shap(self):
        shap.initjs()
        explainer = shap.TreeExplainer(self.forest)
        self.shap_values = explainer.shap_values(self.X_train)

    def summary_plots(self):
        plt.title('ADOPTION')
        shap.summary_plot(self.shap_values[0], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/adoption_summary.png')
        plt.clf()
        plt.title('DIED')
        shap.summary_plot(self.shap_values[1], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/died_summary.png')
        plt.clf()
        plt.title('EUTHANASIA')
        shap.summary_plot(self.shap_values[2], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/euthanasia_summary.png')
        plt.clf()
        plt.title('RETURN TO OWNER')
        shap.summary_plot(self.shap_values[3], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/return_to_owner_summary.png')
        plt.clf()
        plt.title('TRANSFER')
        shap.summary_plot(self.shap_values[4], self.X_train, show=False, plot_size=(16, 6))
        plt.savefig('plots/transfer_summary.png')

    def age_dependency_plot(self):
        shap.dependence_plot('AgeuponOutcome', self.shap_values[0], self.X_train, interaction_index=None, show=False)
        plt.savefig('plots/age_dependency.png')
