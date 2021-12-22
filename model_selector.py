import logging
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings
warnings.filterwarnings('ignore')


class ModelSelector:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/model_selector.log', mode='w')
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
        self.kneigh = KNeighborsClassifier()
        self.dectree = DecisionTreeClassifier(random_state=42)
        self.forest = RandomForestClassifier(random_state=42)
        self.adab = AdaBoostClassifier(random_state=42)
        self.gb = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.make_test_probability_like()
        self.test_kneighbor()
        self.test_dectree()
        self.test_random_forest()
        self.test_adab()
        self.test_xgb()
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

    def logloss(self, y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -(y_true * np.log(y_pred)).sum(axis=1).mean()

    def bayes_search(self, model, param_grid):
        n_iter = 5
        cv = StratifiedKFold(n_splits=n_iter, shuffle=True, random_state=42)
        bsearch = BayesSearchCV(model, param_grid, n_iter=n_iter, scoring='neg_log_loss', cv=cv).fit(
            self.X, self.y)
        self.logger.debug(f"{model}'s best score: {bsearch.best_score_}")
        self.logger.debug(f"{model}'s best parameters: {bsearch.best_params_}")

    def bs_kneighbor(self):
        param_grid = {'n_neighbors': Integer(2, 20),
                      'weights': Categorical(['uniform', 'distance']),
                      'leaf_size': Integer(10, 100)}
        self.bayes_search(self.kneigh, param_grid)

    def test_kneighbor(self):
        self.bs_kneighbor()
        kneigh = KNeighborsClassifier(leaf_size=59, n_neighbors=17)
        kneigh.fit(self.X_train, self.y_train)
        y_pred = kneigh.predict_proba(self.X_test)
        result = self.logloss(self.y_test, y_pred)
        self.logger.debug(f"Kneighbor's result: {result}")

    def bs_dectree(self):
        param_grid = {'criterion': Categorical(['gini', 'entropy']),
                      'splitter': Categorical(['best', 'random']),
                      'max_depth': Integer(10, 200),
                      'min_samples_split': Integer(5, 50),
                      'max_leaf_nodes': Integer(10, 200),
                      }
        self.bayes_search(self.dectree, param_grid)

    def test_dectree(self):
        self.bs_dectree()
        dectree = DecisionTreeClassifier(max_depth=69, min_samples_split=43, splitter='random', random_state=42)
        dectree.fit(self.X_train, self.y_train)
        y_pred = dectree.predict_proba(self.X_test)
        result = self.logloss(self.y_test, y_pred)
        self.logger.debug(f"Decision Tree's result: {result}")

    def bs_random_forest(self):
        param_grid = {'n_estimators': Integer(100, 2000),
                      'criterion': Categorical(['gini', 'entropy']),
                      'max_leaf_nodes': Integer(20, 500),
                      'min_samples_split': Integer(5, 50),
                      }
        self.bayes_search(self.forest, param_grid)

    def test_random_forest(self):
        self.bs_random_forest()
        forest = RandomForestClassifier(criterion='entropy', max_leaf_nodes=83, min_samples_split=10,
                                        n_estimators=173, random_state=42)
        forest.fit(self.X_train, self.y_train)
        y_pred = forest.predict_proba(self.X_test)
        result = self.logloss(self.y_test, y_pred)
        self.logger.debug(f"Random Forest's result: {result}")

    def bs_adab(self):
        param_grid = {'n_estimators': Integer(50, 1000),
                      'learning_rate': Real(0.01, 1, prior='log-uniform')
                      }
        self.bayes_search(self.adab, param_grid)

    def test_adab(self):
        self.bs_adab()
        adab = AdaBoostClassifier(learning_rate=0.01, n_estimators=640, random_state=42)
        adab.fit(self.X_train, self.y_train)
        y_pred = adab.predict_proba(self.X_test)
        result = self.logloss(self.y_test, y_pred)
        self.logger.debug(f"AdaBoost's result: {result}")

    def bs_xgb(self):
        param_grid = {'max_depth': Integer(1, 90),
                      'learning_rate': Real(0.01, 1, prior='log-uniform'),
                      'reg_alpha': Real(0.01, 100),
                      'colsample_bytree': Real(0.2e0, 0.8e0),
                      'subsample': Real(0.2e0, 0.8e0),
                      'n_estimators': Integer(50, 200)}
        self.bayes_search(self.gb, param_grid)

    def test_xgb(self):
        self.bs_xgb()
        gb = xgb.XGBClassifier(colsample_bytree=0.8, learning_rate=0.18, max_depth=10, n_estimators=81,
                               reg_alpha=42, subsample=0.54, eval_metric='mlogloss', random_state=42)
        gb.fit(self.X_train, self.y_train)
        y_pred = gb.predict_proba(self.X_test)
        result = self.logloss(self.y_test, y_pred)
        self.logger.debug(f"XGBoost's result: {result}")
        self.logger.debug("The best model was Random Forest Classifier")
