import json
import os

from src.train import Model_NN


if __name__ == '__main__':

    # LOAD CONFIGURATION ============================================================================

    config_path = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(config_path, 'LyricsGenerator/config.json')) as jf:
        config = json.load(jf)

    root_dir = config['paths']['root_path']

    # INSTANTIATE GENERIC READ OBJECT ===============================================================

    model_obj = Model_NN(config['model_config'],
                         root_dir + config['paths']['data_path'] + '/lyrics_new.txt',
                         root_dir + config['paths']['glove_path'])

    model_obj.load_glove()

    model_obj.data_sequences()

    model_obj.prepare_data()

    model_obj.custom_embedding_layer()

    model_obj.train_model()

    predictions = model_obj.predict(4)
    print(predictions)