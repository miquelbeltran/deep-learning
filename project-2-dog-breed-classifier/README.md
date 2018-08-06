# Dog Breed Classifier

Project for the Udacity Nanodegree: Deep Learning

This is a fork of the original project that includes my own solution. To read the original README.md please follow this link:
https://github.com/udacity/dog-project

## Summary

In this project, I perform the knowledge transfer technique over the existing neural network: [InceptionV3](https://keras.io/applications/#inceptionv3) to classify dog breeds from images.

For comparison, I first create my own CNN from scratch, obtaining a sad accuracy of 7%.

With my neural network on top of InceptionV3, my accuracy score goes up to 81%.

The project also includes simple face recognition using OpenCV Haar Cascades to detect if a picture contain a human face. When the picture contains a human, it will also try to guess the dog breed of this person.

To explore the solution, [take a look at the Notebook](https://github.com/miquelbeltran/dog-project/blob/master/dog_app.ipynb).
