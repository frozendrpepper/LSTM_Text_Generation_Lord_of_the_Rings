![Alt text](https://github.com/frozendrpepper/LSTM_Text_Generation_Lord_of_the_Rings/blob/master/final_result.png)
(Source: OpenFace Github Repository)

# Text Generation Using Stacked LSTM with Lord of the Rings

The goal is to construct a simple stacked LSTM layer that can generate reasonable (a.k.a grammatically sound and coherent)
English sentences based on a given data. Lord of the Rings was chosen as the test dataset as its unique structure would allow
easier assessment of the results (In terms of analyzing overfitting and grammatical structure of the generated sentences).

## Pipeline Summary

1) Face Detection using Object Detection
 * I've used Tensorflow API's built in inception model. This model had better accuracy than the mobilenet model which is the lightest
   model that is provided.
   [Tensorflow API Built In Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
   
2) Face Recognition Deep Nueral Network
 * The basic idea of how face recognition NN works can be seen as follows:
 ![alt text](https://i.ytimg.com/vi/6jfw8MuKwpI/maxresdefault.jpg)
 (Source: Coursera Deeplearning.ai lecture on Siamese Network)
 * The idea is to utilize a pre-trained deep learning model that has been specifically trained to recognize different faces. The output of the deep learning architecture a 128 dimensional vector that represents an encoding of a face image. A distance metric is used to compare different faces and if the distance falls within a certain range, we have matching faces.
 * For this project, the source code provided from Coursera's Deeplearning.ai is used. However, I took out all unnecessary parts 
 and heavily modified the code so as to not give away solution (The course does not want people who have taken the course to 
 release some of the codes since they are solutions to the exercise problems and can cause honor code issue)


## Result

You can see in the main Jupyter Notebook file that the model does an excellent job at detecting all faces in the images
and also recognize my face apart from other faces
 
## Suggestions

I have implemented this on my personal hardware (7th Gen core i7 and GTX 1060 6GB) with a built in camera. The frame rate isn't
the greatest but the model can run at an acceptable frame rate and detect faces (and recognize my own face). OpenCV has built in
methods that can automatcally detect any connected camera to your hardware, input a live stream video and return each frame
as individual image represented as array. 

Additionally, OpenCV has its own face recognition module built into it that uses slightly different approach. This could be a potential future project to compare the performance of different face recognition approaches.

## Useful References

* (https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)
* (https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
* (http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_2__en.htm)
* (https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275)
* (http://cs231n.github.io/neural-networks-3/)

## Acknowledgement

The project was carried as a part of Fall 2018 GRIDS project with Myron Kwan, a fellow MS student in Computer Science.
