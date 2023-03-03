# SDN_PROJ

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
