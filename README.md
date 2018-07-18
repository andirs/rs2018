# Track2Seq
Track2Seq is a Deep Long Short Term Memory network implementation which can be used to generate diverse playlist continuations by predicting one track at a time. The method leverages top-k next-item probabilities to construct a list of recommendations through a semi guided prediction process. In addition the setup shows how title information can be used for playlists when no seed tracks are available. 

## Dependencies
All dependencies are in `requirements.txt` and can be installed i.e. through `pip install -f requirements.txt`

## Data
The network was designed to work with the Million Playlist Dataset (MPD), official website hosted at https://recsys-challenge.spotify.com. 

## Preprocessing
After setting local variables in `src/config.json` pre-processing and training scripts can be executed. Make sure to download the [pre-computed word2vec embeddings](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) for step 6) by [Mikolov et al](https://code.google.com/archive/p/word2vec/). Run the scripts in following order: 

1) `src/a_generate_sequences.py`
2) `src/b_generate_levenshtein_seeds.py`
3) `src/c_generate_w2v_seeds.py`

This will perform the following steps:

1) Generating statistics for all playlists to stratify on
2) Binning and stratification of playlists
3) Splitting of playlists in train, development and test sets
4) Bucketing of development and test sets to match challenge data
5) Turning training playlists in int-sequence and filtering less frequent songs (< 5)
6) Computing seed tracks for 0-seed playlists

The whole process will take a while to compute and requires sufficient memory as well as disk space. Pre-computed files as well as weights will be made available as well. 

## Model
Afterwards the model can be trained by running `src/rnn.py`
