# Financial Fraud Detection

INTRODUCTION

Credit card fraud happens when someone (mostly a fraudster) uses the stolen credit card or the credit card information to make unauthorized purchases. The reason I chose this topic is that this problem is real and so many people are getting scammed because of credit card fraudulent activities. I am going to use Synthetic Financial Dataset for Fraud Detection.

Link for the dataset: https://www.kaggle.com/datasets/ealaxi/paysim1.

The dataset is very large with around 6 million entries and 11 columns. Due to computational limitations, I would be making use of 100,000 entries of the dataset.
The second phase of this project involves developing Learning Vector Quantization Algorithm (LVQ) to classify fraudulent and non-fraudulent activities. LVQ algorithm works on the principle of distance between weight matrix and input vector. As we are concerned with the distance in LVQ, we do not need to scale the data for this algorithm.

There are three folders in this GitHub repo namely, Udbhav-Kush, presentation-and-proposal, and initial_work

Folder named Udbhav-Kush contains two folder inside it namely, code and Individual-Final-Project-Report-Udbhav-Kush

Code folder contains all the code files I have developed for this project.

The main file to run is 'main_file_project.py'. This file imports other supporting files like lvq.py and toolbox.py.

lvq.py is the implementation of LVQ algorithm which is imported by main file to use the class LVQ.

toolbox.py contains some supporting functions of the code like activation_func() and I have defined all the activation functions in that file.

irisLVQ.py was developed to test the implemented LVQ class on the IRIS dataset.

The code folder also contains the updated dataset (just 100k samples from the complete dataset.)

Individual-Final-Project-Report-Udbhav-Kush folder contains the final report of the project in .pdf format.

Presentation-and-proposal contains the project presentation and project proposal.

initial_work folder contains all the code files relevant to the project + preprocessing.py file which was developed to sample 100k values from the original dataset.




