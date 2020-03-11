# Deep Learning with LSTM for Presidential Tweet Generation
The goal of this project is to train a LSTM recurrent neural network on the corpus of President Trumps's tweets and to then use the resulting model to create tweets by autocompleting some seed text. I am processing the tweets in windows of 60 characters and using the next character as the prediction target. I generate training data by iterating through each tweet and sliding the window by 3 characters at a time. This produces training data, which I onehot encode. Currently I am using a LSTM with 256 units, with categorical cross-entropy loss and the Adam optimizer. LSTM leads to a Dense Layer which applies softmax to generated a next character probability distribution. The training will stop early if training loss stops decreasing for 3 epochs. 
# Example outputs
After training the model for 10 epochs, it generates tweets like these:
```python
>>> generate_tweet('house de',model,tokenizer,maxlen=60,num_chars=63,temp=1)
```
>'house deal were all the new merivies when the new hampshire! thank you! #trump2016 #trump2016 http://t.co/h4bqjgokat  @trumpforpresident @realdonaldtrump donald trump we love the hoster is a president was not many place course interview tonight to be a far what we will expert'

```python
>>> generate_tweet('i alone can ',model,tokenizer,maxlen=60,num_chars=63,temp=.5)
```
>'i alone can donald trump hotel and friend of the world to make a trump and see you in president. #trumpforpresident obama is one of the entire with the way to me on some office who would run for president and the plans to have the results in from named a national deal &amp;'
```python
>>> generate_tweet('today i managed ',model,tokenizer,maxlen=60,num_chars=63,temp=.5)
```
>'today i managed him bush strong and we will be a great mind and the decome than even tonight. look forward to have the new year than again in eany #trump2016 #trump2016 #trump2016 #trump2016 great president a great state of the biggest and we are star in the world! not someone'

It appears the model needs more training, or the hyperparameters need more tuning. 
## TODO:
* Make this compatible with Google Colab
* Train model more