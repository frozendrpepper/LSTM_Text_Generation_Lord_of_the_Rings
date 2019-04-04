


# Text Generation Using Stacked LSTM with Lord of the Rings

The goal is to construct a simple stacked LSTM layer that can generate reasonable (a.k.a grammatically sound and coherent)
English sentences based on a given data. Lord of the Rings was chosen as the test dataset as its unique structure would allow
easier assessment of the results (In terms of analyzing overfitting and grammatical structure of the generated sentences).

## Pipeline Summary




## Result

![Alt text](https://github.com/frozendrpepper/LSTM_Text_Generation_Lord_of_the_Rings/blob/master/final_result.png)

If parts of the sentences are from the same passage in the original text, they are highlighted with the same color.
As you can see, three layer model was able to generate a much more stable/complex sentence without overfitting to the dataset.
This overfitting is happening partially due to the fact the dataset used here is relatively small (We had to use a smaller dataset
in the interest of finishing the project and demonstrating).

Unlike typical supervised learning models, it is a bad idea to judge the performance of your model based on accuracy because
higher accuracy means the model is overfitting more and more to the training dataset. 
 
## Suggestions

We could not find a good numeric value that could be assessed with each epoch in our training steps to analyze what would be the 
optimal number of epochs. In fact, the judgment came from trying incremental number of epochs and judging 

## Useful References

* (https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
* (https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
* (http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_2__en.htm)
* (https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275)
* (http://cs231n.github.io/neural-networks-3/)

## Acknowledgement

The project was carried as a part of Fall 2018 GRIDS project with Myron Kwan, a fellow MS student in Computer Science.
