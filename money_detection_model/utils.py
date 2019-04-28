import os
from simple_state_recurrent_model.model import load_saved_model

def get_newest_model():
    model_dir = os.path.join(os.path.dirname(__file__), "models")

    if len(os.listdir(model_dir)) == 0:
        raise Exception("No serialized models found in package instalation.")

    newest_model = max(os.listdir(model_dir))
    return os.path.join(model_dir, newest_model)

money_detection_model = load_saved_model(get_newest_model())

def run_inference(string):
    return money_detection_model.compute_inference(string)
