![Alt text](https://i.ytimg.com/vi/0w43C8jFX2o/maxresdefault.jpg)


# Text Generation Using Stacked LSTM with Lord of the Rings

The goal is to construct a simple stacked LSTM layer that can generate reasonable (a.k.a grammatically sound and coherent)
English sentences based on a given data. Lord of the Rings was chosen as the test dataset as its unique structure would allow
easier assessment of the results (In terms of analyzing overfitting and grammatical structure of the generated sentences).

## Data
![Alt text](https://cdn-images-1.medium.com/max/1200/0*XMW5mf81LSHodnTi.png)

Word embedding is used to prevent use of large sparse vectors as input data.

As already mentioned, one of the Lord of the Rings novel was chosen because it has unique grammar and style that would make it
easier to debug if there are any issues with the model.

Additionally, in the interest of completing the project within a reasonable period of time, we only utilized part of the training
dataset which we have provided here. 

## Model Summary

![Alt text](https://cs.stanford.edu/people/karpathy/recurrentjs/eg.png)

As shown above, we used Keras package with Tensorflow-GPU backend to construct a stacked LSTM model. At each LSTM layer, we set 
return_sequences set to true such that each LSTM passes its output/hidden state onto the next layer of LSTMs. In addition, to minimize
overfitting, Dropout layers with dropout rate of 0.3 are added between the LSTM layers. The final layer is a softmax layer where
the label represents the next word in the text.

![Alt text](https://cdn-images-1.medium.com/max/1600/1*XvUt5wDQA8D3C0wAuxAvbA.png)

The way input to the model is structured is shown above. We used sequential stepping to generate the next word. Basically, we would
fix a number of seed words to generate next letter and the step forward to the second word as the first word of next iteration to do
the same thing until specified sentence length has been met.

## Result

![Alt text](https://github.com/frozendrpepper/LSTM_Text_Generation_Lord_of_the_Rings/blob/master/final_result.png)

The best result was achieved using three layered LSTM model where each LSTM had 700 internal units, dropout ratio of 0.3, 
Word Embedding dimension of 128 and at 50 epochs/Batch size 100.

If parts of the sentences are from the same passage in the original text, they are highlighted with the same color.
As you can see, three layer model was able to generate a much more stable/complex sentence without overfitting to the dataset.
This overfitting is happening partially due to the fact the dataset used here is relatively small (We had to use a smaller dataset
in the interest of finishing the project and demonstrating). As well known, using larger dataset can often lead to preventing
overfitting.

Unlike typical supervised learning models, it is a bad idea to judge the performance of your model based on accuracy because
higher accuracy means the model is overfitting more and more to the training dataset. 
 
## Suggestions

We could not find a good numeric value that could be assessed with each epoch in our training steps to analyze what would be the 
optimal number of epochs. In fact, the judgment came from trying incremental number of epochs and "eyeballing" how good
sentences look. Obviously, this approach would not scale well to a project with larger data and/or more complex model
and this is something we wish to include. 

Most of the reference sources that we have worked with seem to have same issue. No one was able to clearly provide
the metrics they've utilized to judge their model performance. Rather, it seems like it is done purely empirically
and based on personal judgments.

## Useful References

* (https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
* (https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
* (http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_2__en.htm)
* (https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275)
* (http://cs231n.github.io/neural-networks-3/)

## Acknowledgement

The project was carried as a part of Fall 2018 GRIDS project with Myron Kwan, a fellow MS student in Computer Science.
