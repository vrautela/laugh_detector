"""A module/script that continuously records audio and periodically determines whether laughter occurred during chunks of the recording"""
from fastai.learner import load_learner
import logging
from pathlib import Path
import pyaudio
import wave
import subprocess
from pydub import AudioSegment
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from pickle import load
import random

logging.basicConfig(encoding='utf-8', level=logging.INFO)

MODEL_PATH = Path('./model_sklearn_1000_features.pkl')
MODE = 'sklearn' # can be one of 'sklearn' or 'fastai'

# Parameters for the recording
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono channel
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 1024              # Buffer size
RECORD_SECONDS = 3        # Duration of recording (in seconds)
RAW_AUDIO_FILENAME = "raw_audio.wav"  # Output file name

def pad_wav_to_10_seconds(input_wav_path, output_wav_path):
    # Load the audio file
    audio = AudioSegment.from_wav(input_wav_path)

    # Determine the duration of the audio in milliseconds
    audio_duration_ms = len(audio)

    # Calculate the required padding duration to reach 10 seconds (10000 milliseconds)
    target_duration_ms = 10 * 1000
    padding_duration_ms = target_duration_ms - audio_duration_ms

    # If padding is needed, add silence to the end of the audio
    if padding_duration_ms > 0:
        # Round up to ensure the total length reaches exactly 10 seconds
        silence = AudioSegment.silent(duration=padding_duration_ms + 1)
        padded_audio = audio + silence

        # Trim the padded audio to exactly 10 seconds
        padded_audio = padded_audio[:target_duration_ms]
    else:
        padded_audio = audio[:target_duration_ms]

    # Export the padded audio to a new .wav file
    padded_audio.export(output_wav_path, format="wav")

    # Return the length of the padded audio for verification
    return len(padded_audio)


def parse_feature_data(raw_dataset):
    """
    This function takes the raw dataset and produces a table that contains the raw 128x10 features

    Note: we will skip videos that aren't 10 seconds long
    """

    extracted_table = []

    for raw_record in raw_dataset:
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())

        if len(example.feature_lists.feature_list['audio_embedding'].feature) != 10:
            continue

        audio_features_10s = [np.frombuffer(example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],np.uint8).astype(np.float32) for i in range(10)]

        flat_audio_features = [j for sub in audio_features_10s for j in sub]

        extracted_data = flat_audio_features
        extracted_table.append(extracted_data)

    return extracted_table


def main():
    logging.info("Loading laugh detector...")
    learn_inference = None
    if MODE == 'sklearn':
        with open(MODEL_PATH, 'rb') as f:
            learn_inference = load(f)
    elif MODE == 'fastai':
        # load laugh detector into local variable
        learn_inference = load_learner(MODEL_PATH)
    else:
        raise ValueError(f"Invalid mode: {MODE}")
    logging.info("Listening for laughter! Hit Ctrl+C to quit the program.")

    while True:
        # record L seconds of audio and save it to a .wav file
        # pad the .wav file to 10 seconds
        # extract features from the .wav file using vggish_inference_demo
        # turn features into a (1-row) pandas DataFrame
        # call laugh detector on the DataFrame
        # print whether or not laughter was detected




        # Create an interface to PortAudio
        audio = pyaudio.PyAudio()

        # Open a stream for recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        logging.info("Recording...")

        frames = []

        # Record audio in chunks
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        logging.info("Finished recording.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded data as a wave file
        wave_file = wave.open(RAW_AUDIO_FILENAME, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

        logging.info(f"Saved raw audio to {RAW_AUDIO_FILENAME}")

        PADDED_AUDIO_FILENAME = "padded_output.wav"
        pad_wav_to_10_seconds(RAW_AUDIO_FILENAME, PADDED_AUDIO_FILENAME)

        logging.info(f"Saved padded audio to {PADDED_AUDIO_FILENAME}")

        t0=time.time()
        TFRECORD_FILENAME = "padded_output_features.tfrecord"
        # extract features from the .wav file using vggish_inference_demo
        subprocess.run(
            [
                "python",
                "audioset/vggish_inference_demo.py",
                "--wav_file",
                PADDED_AUDIO_FILENAME,
                "--tfrecord_file",
                TFRECORD_FILENAME,
                "--checkpoint",
                "audioset/vggish_model.ckpt",
                "--pca_params",
                "audioset/vggish_pca_params.npz",
                # ">",
                # "/dev/null",
            ]
        )
        t1=time.time()
        logging.info(f"Feature extraction took {t1-t0} seconds")


        # t0=time.time()
        filenames = [TFRECORD_FILENAME]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        features = parse_feature_data(raw_dataset)
        # t1=time.time()

        # logging.info(f"Feature parsing took {t1-t0} seconds")

        np_features = np.array(features)


        if MODE == 'sklearn':
            preds = learn_inference.predict(np_features)
            if preds[0] == 0:
                print("No laughter!")
            else:
                print("Laughter!")
                laugh_audios = ["crowd_laugh_1.wav", "friends_laugh.wav"]
                audio_path = random.choice(laugh_audios)
                subprocess.run(
                    [
                        "afplay",
                        # "crowd_laugh_1.wav"
                        audio_path,
                    ]
                )
        elif MODE == 'fastai':
            features_dict = {f'x_{i}': np_features[:, i] for i in range(1280)}
            features_df = pd.DataFrame.from_dict(features_dict)

            dl = learn_inference.dls.test_dl(features_df)
            preds = learn_inference.get_preds(dl=dl)

            print(f"Model preds: {preds[0]}")

            for row in preds[0]:
                if row[0] > row[1]:
                    print("No laughter!")
                else:
                    print("Laughter!")









if __name__ == '__main__':
    main()
