from data_processor import DataProcessor
from exploratory_data_analysis import DataAnalyser
from model_selector import ModelSelector
from final_model_trainer import ModelTrainer

data_processor = DataProcessor()
data_processor.start()

data_analyser = DataAnalyser()
data_analyser.start()

model_selector = ModelSelector()
model_selector.start()

final_model_trainer = ModelTrainer()
final_model_trainer.start()
