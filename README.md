This is a compression algorithm for images that contain a single important object or Person.
My algorithm essentially applies lossy averaging on unimportant parts of the image to preserve the important parts from being messed with and stay almost lossless.
So, in order to achieve that I found out about a deep learning model that detects certain objects in an image and bounds them by a rectangle. 
That was exactly what I wanted so I utilized this model and used it to determine the bounds for me. 
I take only the biggest object detected by the model and store its pixels untouched and I apply an averaging algorithm around that object. 

For running the project you should run the following command in the terminal.


python main.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
