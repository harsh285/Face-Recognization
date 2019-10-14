# Face_Recognization
to develop a face detection system using FaceNet and an SVM classifier to identify people from photographs.


# Prerequisites

  ### Download From Google Drive
  
   Download pre-trained Keras FaceNet model and Kaggle's Dataset For Face Recognition.
         
   * [Facenet Model And Data](https://drive.google.com/open?id=1ht3M3g3ndYLKIbFcxFXe7AGTlwve33k9)
  
# Description  
  
 ### `Pre_Process.py`
 
 Running the example may take a moment.

First, all of the photos in the ‘train‘ dataset are loaded, then faces are extracted, resulting in 108 samples with square face input and a class label string as output. Then the ‘val‘ dataset is loaded, providing 30 samples that can be used as a test dataset.

Both datasets are then saved to a compressed NumPy array file called ‘faces-dataset.npz‘ that is about three megabytes and is stored in the current working directory.

This dataset is ready to be provided to a face detection model.
 
### `Pre-Trained FaceNet Model`
 
 A face embedding is a vector that represents the features extracted from the face. This can then be compared with the vectors generated for other faces. For example, another vector that is close (by some measure) may be the same person, whereas another vector that is far (by some measure) may be a different person.
The classifier model that we want to develop will take a face embedding as input and predict the identity of the face. The FaceNet model will generate this embedding for a given image of a face.
The FaceNet model can be used as part of the classifier itself, or we can use the FaceNet model to pre-process a face to create a face embedding that can be stored and used as input to our classifier model. This latter approach is preferred as the FaceNet model is both large and slow to create a face embedding.
We can, therefore, pre-compute the face embeddings for all faces in the train and test (formally ‘val‘) sets in our Faces Dataset.

 ### `embedding.py`
 
Running the example reports progress along the way.

We can see that the face dataset was loaded correctly and so was the model. The train dataset was then transformed into 108 face embeddings, each comprised of a 128 element vector. The 30 examples in the test dataset were also suitably converted to face embeddings.

The resulting datasets were then saved to a compressed NumPy array that is about 50 kilobytes with the name ‘faces-embeddings.npz‘ in the current working directory.
 
 ### `Classification.py`
 
 Next, the data requires some minor preparation prior to modeling.

First, it is a good practice to normalize the face embedding vectors. It is a good practice because the vectors are often compared to each other using a distance metric.

In this context, vector normalization means scaling the values until the length or magnitude of the vectors is 1 or unit length. This can be achieved using the Normalizer class in scikit-learn. It might even be more convenient to perform this step when the face embeddings are created in the previous step.

It is common to use a Linear Support Vector Machine (SVM) when working with normalized face embedding inputs. This is because the method is very effective at separating the face embedding vectors. We can fit a linear SVM to the training data using the SVC class in scikit-learn and setting the ‘kernel‘ attribute to ‘linear‘. We may also want probabilities later when making predictions, which can be configured by setting ‘probability‘ to ‘True‘.

Than we need to select a random example from the test set, then get the embedding, face pixels, expected class prediction, and the corresponding name for the class.
 
A different random example from the test dataset will be selected each time the code is run.



##  `Triplet loss`

The model is a deep convolutional neural network trained via a triplet loss function that encourages vectors for the same identity to become more similar (smaller distance), whereas vectors for different identities are expected to become less similar (larger distance). The focus on training a model to create embeddings directly (rather than extracting them from an intermediate layer of a model) was an important innovation in this work.

Triplet loss is a loss function for artificial neural networks where a baseline (anchor) input is compared to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized.

It is often used for learning similarity for the purpose of learning embeddings, like word embeddings and even thought vectors, and metric learning.

The loss function can be described using a Euclidean distance function.




         
         
    
