"""A module/script that continuously records audio and periodically determines whether laughter occurred during chunks of the recording"""
from fastai.learner import load_learner
import logging
from pathlib import Path

# TODO: make this a relative path
# MODEL_PATH = Path('/home/vedrau/work/laugh_detector/model_export_tabular_data_100_features.pkl')
MODEL_PATH = Path('./model_export_tabular_data_100_features.pkl')
L = 2 # the length of each recording (in seconds)

def main():
    # load laugh detector into local variable
    learn_inference = load_learner(MODEL_PATH)
    logging.info("Listening for laughter! Hit Ctrl+C to quit the program.")

    while True:
        # record L seconds of audio and save it to a .wav file
        # pad the .wav file to 10 seconds
        # extract features from the .wav file using vggish_inference_demo
        # turn features into a (1-row) pandas DataFrame
        # call laugh detector on the DataFrame
        # print whether or not laughter was detected

if __name__ == '__main__':
    main()