# Deep Learning with LSTM for Presidential Tweet Generation
The goal of this project is to train a LSTM recurrent neural network on the corpus of President Trumps's tweets and to then use the resulting model to create tweets by autocompleting some seed text. I am processing the tweets in windows of 60 characters and using the next character as the prediction target. I generate training data by iterating through each tweet and sliding the window by 3 characters at a time. This produces training data, which I onehot encode. Currently I am using a LSTM with 256 units. LSTM leads to a Dense Layer which applies softmax to generated a next character probability distribution. 
## TODO:
* Make this compatible with Google Colab
* Add examples of the tweets generated