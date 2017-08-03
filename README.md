# Classify dog breeds
This project was submitted as a part of Udacity Machine Learning Engineer Nanodegree program. 

## Overview
Given an image of a dog, the algorithm will generate an estimate of the dog's breed. 
If supplied an image of a human, the code will identify the resembling dog breed.
To run the code, run the following on th eterminal. 

```
python dog_app.py --img path_to_img.jpg
```



## Examples
![Alt text](https://github.com/JennyLeeStat/Guess_dog_breed/blob/master/images/rei_res.png)

I ran the algorithm on the picture of my shiba inu dog. As 'shiba inu' was not included in the 133 labels we trained 
the inception bottleneck features, the results were bound to be inaccurate. 
He was estimated a mix of canaan dog, Norweisian buhund, Finnish spitz, and akita. This brings up the first idea of improvement. 
There are more than 300 dog breeds and our model can recognize only 1/3 of them. 
By acquiring and building dataset in addition to what we already have, we could curve this kind of error

## Improvements


1. One significant shortcoming this algorithm has is that 
it cannot distinguish if the input image contains more than two different dog breeds. 
I could avoid the similar problem if there are more than one people in the input image by recognizing and cropping faces,
 then providing individual guesses. 
 In order to implement this with images with multiple dogs, we need a more sophisticated dog face detector, 
 such as [this](http://blog.dlib.net/2016/10/hipsterize-your-dog-with-deep-learning.html).

2. I found the running the algorithm on my labtop cpu was pretty slow. 
Google recently realsed the mobilenet faster and light object detection API. 
I will retrain our dog images with [mobilenet](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html) 
if it could improve the runtime and accuracy.


## Environment
The conda environment file can be found at environment.yml. 


## Reference
- https://github.com/udacity/dog-project

