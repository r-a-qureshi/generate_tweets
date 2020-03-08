import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import layers
from keras.utils import to_categorical

# read the data from the http://www.trumptwitterarchive.com/archive
tweets = pd.read_csv('trump_tweets.csv')
tweets = tweets['text'].tolist()

# set up 
tokenizer = Tokenizer(char_level=True,lower=False,filters="")
# use most recent 10,000 tweets
# for some reason using all tweets causes an error in the tokenizer
tokenizer.fit_on_texts(tweets[:10000])
mat = tokenizer.texts_to_matrix(tweets,mode='binary')
seq = tokenizer.texts_to_sequences(tweets[:10000])
seq = sequence.pad_sequences(seq,maxlen=280)
# find out the size of characters
num_chars = max([max(i) for i in seq])
seq = to_categorical(seq,num_chars+1)
# get training and testing sequences

# X = [i[:-1] for i in seq]
# y = [i[1:] for i in seq]


# build LSTM model
model = keras.models.Sequential()
model.add(layers.LSTM(128,input_shape=(280,len(seq))))
model.add(layers.Dense(len(seq),activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

# training the model
# model.fit(X,y,batch_size=128,epochs=10)