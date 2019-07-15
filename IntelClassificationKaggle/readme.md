Link to the blog for image classification - <a href='https://medium.com/@sanchittanwar75/image-classification-tutorials-in-pytorch-transfer-learning-19ebc329e200?postPublishedType=repub'>
link</a>

# Class activation map
I tried to implement class activation map in this problem. The results are good. The code for generating class activation maps can be seen [here](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/cam.py).

Results:

Image             |       Predicted Class           | Class Activation Map
:------------------------------:|:---------------------------:|:-------------------------------:
![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/1.png)| Forest |  ![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/cam1.png)
![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/2.png)| Building |  ![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/cam2.png)
![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/3.png)| Mountain |  ![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/cam3.png)
![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/4.png)| Sea |  ![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/cam4.png)
![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/5.png)| Forest |  ![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/cam5.png)
![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/6.png)| Mountain |  ![](https://github.com/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/results/cam6.png)

We can see in 4th image the model predicted wrong image but as we can see in class activation map model was seeing at the water when it predicted sea. 
