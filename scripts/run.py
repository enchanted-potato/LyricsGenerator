import json
import os
import sys

# Add project root to Python path so we can import lyricsgenerator
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lyricsgenerator.train import Model_NN

if __name__ == "__main__":
    # LOAD CONFIGURATION ============================================================================

    # Get project root directory (parent of scripts directory)
    config_path = os.path.join(project_root, "config.json")
    
    with open(config_path) as jf:
        config = json.load(jf)

    root_dir = config["paths"]["root_path"]

    # INSTANTIATE GENERIC READ OBJECT ===============================================================

    model_obj = Model_NN(
        config["model_config"],
        root_dir + config["paths"]["data_path"] + "/lyrics_new.txt",
        root_dir + config["paths"]["glove_path"],
    )

    model_obj.load_glove()

    model_obj.data_sequences()

    model_obj.prepare_data()

    model_obj.custom_embedding_layer()

    model_obj.train_model()

    model_obj.predict_model()

    predictions = model_obj.generate_lyrics(4)
    print(predictions)
