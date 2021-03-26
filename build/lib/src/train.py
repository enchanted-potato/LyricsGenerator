from __future__ import print_function, division
from builtins import range, input

import os
import numpy as np
import pickle

from keras.models import Model, model_from_json
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD

try:
  import keras.backend as K
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
except:
  pass


class Model_NN(object):

    def __init__(self, configurations, lyrics_path, glove_path, data_path):

        for attr in configurations.keys():
            setattr(self, attr, configurations[attr])

        self.lyrics_path = lyrics_path
        self.glove_path = glove_path
        self.data_path = data_path

    def load_lyrics(self):
        data = []
        for line in open(self.lyrics_path, encoding='utf8'):
          line = line.rstrip()
          if line!='' and line!='<|endoftext|>':
            data.append(line)

        return

    def load_glove(self):
        # load in pre-trained word vectors
        print('Loading word vectors...')
        word2vec = {}
        with open(os.path.join(self.glove_path), encoding='utf8') as f:
          # is just a space-separated text file in the format:
          # word vec[0] vec[1] vec[2] ...
          for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
        print('Found %s word vectors.' % len(word2vec))

        self.word2vec = word2vec

        return

    def clean_text(self):

        for line in open(self.lyrics_path, encoding='utf8'):
            line = line.rstrip()
            if not line:
                continue


    def data_sequences(self):
        # load in the data
        input_texts = []
        target_texts = []
        for line in open(self.lyrics_path, encoding='utf8'):
            if line != '' and '<|endoftext|>' not in line:
                line = line.rstrip()
                if not line:
                    continue

                input_line = '<sos> ' + line
                target_line = line + ' <eos>'

                input_texts.append(input_line)
                target_texts.append(target_line)

        all_lines = input_texts + target_texts

        # convert the sentences (strings) into integers
        self.tokenizer = Tokenizer(num_words=self.MAX_VOCAB_SIZE, filters='') # filters='' ensures that special characters like <sos> are not filtered out
        self.tokenizer.fit_on_texts(all_lines)
        self.input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        self.target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        return


    def prepare_data(self):

        # find max seq length
        max_sequence_length_from_data = max(len(s) for s in self.input_sequences)
        print('Max sequence length:', max_sequence_length_from_data)


        # get word -> integer mapping (dictionary)
        word2idx = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word2idx))
        assert('<sos>' in word2idx)
        assert('<eos>' in word2idx)

        try:
            word2idx_file = open(self.data_path + '/word2idx.pickle', 'wb')
            pickle.dump(word2idx, word2idx_file)
            word2idx_file.close()

        except:
            print("Could not save word2idx")


        # pad sequences so that we get a N x T matrix
        max_sequence_length = min(max_sequence_length_from_data, self.MAX_SEQUENCE_LENGTH)
        input_sequences = pad_sequences(self.input_sequences, maxlen=max_sequence_length, padding='post')
        target_sequences = pad_sequences(self.target_sequences, maxlen=max_sequence_length, padding='post')
        print('Shape of data tensor:', input_sequences.shape)



        # prepare embedding matrix
        print('Filling pre-trained embeddings...')
        num_words = min(self.MAX_VOCAB_SIZE, len(word2idx) + 1)
        embedding_matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for word, i in word2idx.items():
          if i < self.MAX_VOCAB_SIZE:
            embedding_vector = self.word2vec.get(word)
            if embedding_vector is not None:
              # words not found in embedding index will be all zeros.
              embedding_matrix[i] = embedding_vector

        # one-hot the targets (can't use sparse cross-entropy)
        one_hot_targets = np.zeros((len(input_sequences), max_sequence_length, num_words))
        for i, target_sequence in enumerate(target_sequences):
          for t, word in enumerate(target_sequence):
            if word > 0:
              one_hot_targets[i, t, word] = 1

        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.one_hot_targets = one_hot_targets
        self.embedding_matrix = embedding_matrix
        self.num_words = num_words
        self.max_sequence_length = max_sequence_length
        self.word2idx = word2idx

        # reverse word2idx dictionary to get back words
        # during prediction
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        return max_sequence_length


    def custom_embedding_layer(self):

        # load pre-trained word embeddings into an Embedding layer
        self.embedding_layer = Embedding(
          self.num_words,
          self.EMBEDDING_DIM,
          weights=[self.embedding_matrix],
          # trainable=False
        )

        return

# TRAIN ----------------------------------------------------------------------------------------------------------------

    def train_model(self):

        print('Building model...')

        # create an LSTM network with a single LSTM
        input_ = Input(shape=(self.max_sequence_length,))
        self.initial_h = Input(shape=(self.LATENT_DIM,))
        self.initial_c = Input(shape=(self.LATENT_DIM,))
        x = self.embedding_layer(input_)
        self.lstm = LSTM(self.LATENT_DIM, return_sequences=True, return_state=True)
        x, _, _ = self.lstm(x, initial_state=[self.initial_h, self.initial_c]) # don't need the states here
        self.dense = Dense(self.num_words, activation='softmax')
        output = self.dense(x)

        model = Model([input_, self.initial_h, self.initial_c], output)
        model.compile(
          loss='categorical_crossentropy',
          # optimizer='rmsprop',
          optimizer=Adam(lr=0.01),
          # optimizer=SGD(lr=0.01, momentum=0.9),
          metrics=['accuracy']
        )

        print('Training model...')
        z = np.zeros((len(self.input_sequences), self.LATENT_DIM))
        r = model.fit(
          [self.input_sequences, z, z],
          self.one_hot_targets,
          batch_size=self.BATCH_SIZE,
          epochs=self.EPOCHS,
          validation_split=self.VALIDATION_SPLIT
        )

        return




    def predict_model(self):

        # make a sampling model
        print('Making a sampling model...')
        input2 = Input(shape=(1,))  # we'll only input one word at a time
        x = self.embedding_layer(input2)
        x, h, c = self.lstm(x, initial_state=[self.initial_h, self.initial_c])  # now we need states to feed back in
        output2 = self.dense(x)
        self.sampling_model = Model([input2, self.initial_h, self.initial_c], [output2, h, c])

        # serialize model to JSON
        model_json = self.sampling_model.to_json()
        with open(self.data_path + "/model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.sampling_model.save_weights(self.data_path + "/model.h5")
        print("Saved model to disk")

        return

    @staticmethod
    def sample_line(config, max_sequence_length):

        data_path = config['paths']['root_path'] + config['paths']['data_path']

        # load the model from pickle file
        json_file = open(data_path + '/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        sampling_model = model_from_json(loaded_model_json)

        # load weights into new model
        sampling_model.load_weights(data_path + '/model.h5')
        print("Loaded model from disk")

        # from config
        LATENT_DIM = config['model_config']['LATENT_DIM']
        max_sequence_length = max_sequence_length

        # load word2idx dictionary
        with open(data_path + '/word2idx.pickle', 'rb') as handle:
            word2idx = pickle.load(handle)

        idx2word = {v: k for k, v in word2idx.items()}

        # initial inputs
        np_input = np.array([[word2idx['<sos>']]]) # 1x1 input with just <sos>
        h = np.zeros((1, LATENT_DIM)) # same as what we used in training for consistency
        c = np.zeros((1, LATENT_DIM)) # same as what we used in training for consistency

        # so we know when to quit
        eos = word2idx['<eos>']

        # store the output here
        output_sentence = []

        for _ in range(max_sequence_length):
            o, h, c = sampling_model.predict([np_input, h, c]) # o is a list of word probabilities for the next word

            # print("o.shape:", o.shape, o[0,0,:10])
            # idx = np.argmax(o[0,0])
            probs = o[0, 0]
            if np.argmax(probs) == 0:
                print("wtf")
            probs[0] = 0 # make <sos> probability 0
            probs /= probs.sum()
            idx = np.random.choice(len(probs), p=probs)
            if idx == eos:
                break

            # accuulate output
            output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx)) # if the index is not in dictionary you get WTF

            # make the next input into model
            np_input[0, 0] = idx

        return ' '.join(output_sentence)

    @staticmethod
    def generate_lyrics(lines, config, max_sequence_length):

        print('Generating predictions...')

        predictions = []
        for _ in range(lines):
            predictions.append(Model_NN.sample_line(config, max_sequence_length))

        print(predictions)

        return predictions