# Invisiblur

Introduction: This is a Flask project for processing videos to anonymize faces. This project requires certain dependencies to be installed before it can be run. This document provides a step-by-step guide on how to install the dependencies and run the project.

Installation:

1.  Install Python 3.x on your machine.
2.  Clone or download the project from GitHub.
3.  Open the terminal or command prompt and navigate to the project directory.
4.  Create a virtual environment by running the following command:
    -   Windows: `python -m venv venv`
    -   macOS/Linux: `python3 -m venv venv`
5.  Activate the virtual environment by running the following command:
    -   Windows: `venv\Scripts\activate`
    -   macOS/Linux: `source venv/bin/activate`
6.  Install the dependencies by running the following command:
    -   `pip install -r requirements.txt`

Running the project:

1.  Make sure that the virtual environment is activated.
2.  Navigate to the project directory in the terminal or command prompt.
3.  Run the following command to start the server:
    -   `python run.py`
4.  Open a web browser and navigate to <http://localhost:5000/>
5.  Upload a video to the webpage and click on the 'Process Video' button.
6.  The processed video will be saved in the folder containing the original video.

Conclusion: This Flask project can be used to anonymize faces in videos. Follow the above steps to install the dependencies and run the project. For any issues or queries, please refer to the project documentation or contact the project contributors.

We are using the concept of convoluton to make the faces of humans invisible, we follow the following steps:


1) We first convert the RGB picture into a greyscale ( from 3X3 to 1X1 for lesser complexity of computation )
2) We then create a mask that we are going to use for creating a blur in the image.
3)  We traverse through each pixel and travel 6(arbitary value) pixels in all four directions and average out the greyscale value to obtain 1 value. This will now be used as a reference ( this is where convolution is being done )
4) We then take the entire video frameby frame and use opencv to detect a face. It gives us a grid pointing to it's edge coordinates
5) we then take the grid of the face and convolute it the same way that we did earlier ( the 6X6 grid process )
6) we then compare each pixel value to the initial(blurred) value of the grid ( the convoluted one ). 
7) If we see a change that exceeds the threshold ( set arbirtarily by us ) we replace that pixel ONLY, with the orignal picture pixel. We need the threshold in order to compensate for noise in the frame.
8) Once this happens, only the face is turned invisible and hence the video is perfectly anonymised.


Using invisiblur, a user can in no way deblur this image as we are essentially erasing data. A normal deblurring software would take a blurred image and using AI tools would analyse each blurred pixel and make an educated guess on what could have been the initial image, all that is not possible in our case as there is no image as such in the processed video.

