Follow the instructions here: https://github.com/tensorflow/models/tree/master/research/audioset/vggish

Specifically note the part about downloading the VGGish model checkpoint and Embedding PCA parameters. Be sure to run `python vggish_smoke_test.py` to ensure there are no errors with your VGGish setup.


To run live inference:

- `python`
- `from live_inference import main as m`
- `m()`


Notes:

Make sure to get the `pyaudio` installation correct.

`pyaudio` doesn't seemt to work on WSL2
