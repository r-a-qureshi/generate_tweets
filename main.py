import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import layers
from keras.utils import to_categorical
import numpy as np
from itertools import count, cycle

NUM_CHARS = 63

# read the data from the http://www.trumptwitterarchive.com/archive
tweets = pd.read_csv('trump_tweets.csv')
tweets.dropna(subset=['text'],inplace=True)
tweets['text'] = tweets['text'].str.lower()
tweets = tweets['text'].tolist()


# set up 
tokenizer = Tokenizer(num_words=None,char_level=True,lower=True,oov_token='NA')
tokenizer.fit_on_texts(tweets)
seq = tokenizer.texts_to_sequences(tweets)
seq = list(filter(lambda x: max(x)<63,seq))
seq = sequence.pad_sequences(seq,maxlen=200,truncating='post',padding='pre')
char_map = dict(map(reversed,tokenizer.word_index.items()))
seq_gen = cycle(seq)
def data_gen(data,maxlen=60,step=3,num_chars=63):
    for twt in data:
        for i in range(0,200-maxlen,step):
            yield(
                to_categorical(twt[i:i+maxlen],num_chars).reshape((1,maxlen,num_chars)),
                to_categorical(twt[i+maxlen],num_chars),
            )

def batch_gen(dgen,batch_size,maxlen=60,num_chars=63):
    while True:
        X = np.zeros((batch_size,maxlen,num_chars))
        y = np.zeros((batch_size,num_chars))
        for i in range(batch_size):
            X[i],y[i] = next(dgen)
        yield(X,y)
            
# build LSTM model
model = keras.models.Sequential()
model.add(layers.LSTM(128,input_shape=(60,NUM_CHARS)))
model.add(layers.Dense(NUM_CHARS,activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

#prepare generators
dgen = data_gen(seq_gen)
bgen = batch_gen(dgen,128)


def reweight_distribution(original,temp=.5):
    dist = np.log(original) / temp
    dist = np.exp(dist)
    return(dist/np.sum(dist))

def sample(preds,temp=.5):
    preds = np.asarray(preds).astype('float64')
    preds = reweight_distribution(preds,temp)
    probs = np.random.multinomial(1,preds.reshape(-1),1).reshape(-1)
    return(np.argmax(probs))


#make a prediction
def generate_tweet(seed,model,tokenizer,tweet='',maxlen=60,num_chars=156,temp=.5):
    """Recursive function to generate tweets by iteratively predicting the next
    character"""
    if tweet == '':
        tweet = seed
    seed = tokenizer.texts_to_sequences([seed])
    seed = sequence.pad_sequences(
        seed,
        maxlen=maxlen,
        truncating='pre',
        padding='pre'
    )
    seed = to_categorical(seed,num_chars)
    prob = model.predict(seed)
    next_char = sample(prob,temp=temp)
    # next_char = prob.argmax()
    reverse_word_map = dict(map(reversed,tokenizer.word_index.items()))
    next_char = reverse_word_map[next_char]
    tweet += next_char
    if len(tweet) == 280:
        tweet = ' '.join(tweet.split(' ')[:-1])
        return(tweet)
    else:
        return(
            generate_tweet(
                tweet[-maxlen:],model,tokenizer,tweet,maxlen,num_chars
            )
        )

