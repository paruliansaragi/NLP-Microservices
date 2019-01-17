import sys, os, re, csv, codecs, numpy as np, pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove.6B.zip  

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

#df = pd.read_csv('./data/mbti_1.csv')

#le = preprocessing.LabelEncoder()
#le.fit(df.type)

#labels = le.transform(df.type)

#df['Label'] = labels

#trn, test = train_test_split(df, test_size=0.1)
#list_sentences_train = trn["posts"].fillna("_na_").values
#y = trn["Label"].values
#list_sentences_test = test["posts"].fillna("_na_").values
#tokenizer = Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(list(list_sentences_train))
#list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
#X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
#X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
#y_train = keras.utils.to_categorical(trn.Label, len(df.Label.unique()))
#path = './'
#EMBEDDING_FILE=f'{path}glove.6B.50d.txt'
#def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

#all_embs = np.stack(embeddings_index.values())
#emb_mean,emb_std = all_embs.mean(), all_embs.std()

#word_index = tokenizer.word_index
#nb_words = min(max_features, len(word_index))
#embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#for word, i in word_index.items():
#    if i >= max_features: continue
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#inp = Input(shape=(maxlen,))
#x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
#x = GlobalMaxPool1D()(x)
#x = Dense(50, activation="relu")(x)
#x = Dropout(0.1)(x)
#x = Dense(16, activation="sigmoid")(x)
#model = Model(inputs=inp, outputs=x)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X_t, y_train, batch_size=32, epochs=5, validation_split=0.1);

