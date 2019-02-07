# Microfacial-exp
Deep learning approach on a classification of emotions based on micro expressions.

Breaking down facial micro-expression is a relatively new research theme with much-developing interests as of late. The principle explanation behind naturally distinguishing the micro-expression is that micro- expression is a critical emotional sign for different real-life applications. Micro-expression is a form of non-verbal communication that unconsciously reveals the true sentiments of a person

I have used various Deep Learning techniques to create a model able to classify a image sequence/video of human faces into one of the six emotions categories based on micro expressions. Presently most approaches use single images to classify emotions based on macro-expression,but they don't consider the dynamics of facial expressions and the minute details that are micro-expression. This approaches considers both these aspects


# dataset 
Micro facial expression recognition is relatively new so not many databases are available as such. I have used in this project CASME dataset. It is a well structured dataset with frame sequences of different people that were asked to suppress there emotions thus the only expressions were the micro expression that are not controllable by consciousness. The six emotion categories are
1. disgust 2. fear 3. happiness 4. repression 5. sadness 6. surprise

# Why need this
Real time emotion recognition is very practical way of detecting potentially harmful behaviour and is a handy way to know the real state of mind of a person. Other approaches require physical testing of the man while it can be done without need of any extra machines or testing. 

# Challenges 
It is not straightforward to recognise the genuine emotion shown on one’s face. Thus recognising micro-expressions is beneficial in our daily life as we can read if someone is attempting to cover his/her feeling or trying to deceive you

# My Approach 
Accurate classification of such emotions on the basis of micro-expressions required both learning of spatial and temporal features.
Out of the many approaches for spatio-temporal learning 3D CNN was found to be most accurate. 3D CNN uses 3D convolution layers of the shape (no.of frames/samples, width. height, channels) this allows us to set a sequence of frames as a single sample and train on that. 3D pooling technique was used for combining all of the learnt features.
As the movement of facial muscles is very minute in this case a method of seperating region of movement was required. I considered two approaches one was the optical flow and the other one was saliency maps. Both use some measure of difference between the frames to find and intensify the region of most change/movement. This created a new dataset of images of region of interest
I tried many network including single stream and multiple streams. What worked best was a two stream 3D-CNN network one stream for normal/raw images and one for the saliency map dataset. Both were required together, the saliency map images helped in learning motion data and normal images helped in learning spatial features of face. Both were concatenated together and the 3D CNN network simulated learning from the sequence, changes in spatio-temporal features during a micro-expression video.The network diagram is as shown below.


