# Track2Seq
Track2Seq is a Deep Long Short Term Memory network implementation which can be used to generate diverse playlist continuations by predicting one track at a time. The method leverages top-k next-item probabilities to construct a list of recommendations through a semi guided prediction process. In addition thie setup shows how title information can be used for playlists when no seed tracks are available. 

## Preprocessing
After setting local variables that point to the data, run `src/generate_sequences.py` to generate the basic data structures. The script will run through the following steps:

1) Generating statistics for all playlists to stratify on
2) Binning and stratification of playlists
3) Splitting of playlists in train, development and test sets
4) Bucketing of development and test sets to match challenge data
5) Turning training playlists in int-sequence and filtering less frequent songs (< 5)
6) Computing seed tracks for 0-seed playlists

This process might take a while and requires sufficient memory as well as disk space.

## Model
Afterwards you are able to train the model by running `src/rnn.py`
