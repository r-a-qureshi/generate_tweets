import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import layers
from keras.utils import to_categorical
import numpy as np

# read the data from the http://www.trumptwitterarchive.com/archive
tweets = pd.read_csv('trump_tweets.csv')
tweets = tweets['text'].tolist()

# set up 
tokenizer = Tokenizer(char_level=True,lower=True)
# use most recent 10,000 tweets
# for some reason using all tweets causes an error in the tokenizer
tokenizer.fit_on_texts(tweets[:10000])
# mat = tokenizer.texts_to_matrix(tweets,mode='binary')
seq = tokenizer.texts_to_sequences(tweets[:10000])
seq = sequence.pad_sequences(seq,maxlen=280)
# generate training data and targets from sequences
def get_data_from_seq(tweets,maxlen=40,step=3,max_chars=68):
    """Encode tweets"""
    tweet_data = []
    target = []
    for tweet in tweets:
        for i in range(0,280-maxlen,step):
            tweet_data.append(to_categorical(tweet[i:i+maxlen],max_chars))
            target.append(to_categorical(tweet[i+maxlen],max_chars))
    return(tweet_data,target)
# find out the size of characters
num_chars = max(tokenizer.word_index.values())+1
X,y = get_data_from_seq(seq,step=20,max_chars=num_chars)
X = np.dstack(X)
shape = X.shape
X = X.reshape(shape[2],shape[0],shape[1])
y = np.vstack(y)
# seq = to_categorical(seq,num_chars+1)
# X = to_categorical(X,num_chars+1)
# y = to_categorical(y,num_chars+1)
# get training and testing sequences

# X = [i[:-1] for i in seq]
# y = [i[1:] for i in seq]


# build LSTM model
model = keras.models.Sequential()
model.add(layers.LSTM(128,input_shape=(40,num_chars)))
model.add(layers.Dense(num_chars,activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

# training the model
# model.fit(X,y,batch_size=128,epochs=10)

def reweight_distribution(original,temperature=.5):
    dist = np.log(original) / temperature
    dist = np.exp(dist)
    return(dist/np.sum(dist))

def sample(preds,temperature=.5):
    preds = np.asarray(preds).astype('float64')
    preds = reweight_distribution(preds,temperature)
    probs = np.random.multinomial(1,preds,1)
    return(np.argmax(probs)+1)


#make a prediction
def generate_tweet(seed,model,tokenizer,tweet='',maxlen=40,num_chars=156):
    if tweet == '':
        tweet = seed
    seed = tokenizer.texts_to_sequences([seed])
    seed = sequence.pad_sequences(seed,maxlen=maxlen)
    seed = to_categorical(seed,num_chars)
    prob = model.predict(seed)
    next_char = sample(prob)
    reverse_word_map = dict(map(reversed,tokenizer.word_index.items()))
    next_char = reverse_word_map[next_char]
    tweet += next_char
    if len(tweet) == 280:
        return(tweet)
    else:
        return(generate_tweet(tweet[-maxlen:],model,tokenizer,tweet,maxlen,num_chars))


