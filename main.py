import sys

from parser import SensorParser
from sensor_preprocess import SensorPreprocessor
from predictor import CognitiveHealthPredictor

def start():
    #parser = SensorParser()
    #parser.start_parsing()

    preprocessor = SensorPreprocessor()
    preprocessor.start_preprocessing()

    predictor = CognitiveHealthPredictor()
    predictor.cognitive_health_prediction()


if __name__ == "__main__":
    start()

