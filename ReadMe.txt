------------------
Usage Instructions
------------------
This program was create on Windows.
From the command line, PredictRSA.py takes two command line arguements.
The first is the model as a pickle file, the second is the .fasta file to be predicted.
The model is provided, it is the RSAmodel.pkl file.

To run the program, from the command line type:		python PredictRSA.py RSAmodel.pkl [filename].fasta
Note: the .fasta file should be in the same directory as PredictRSA.py

The program will create a file in the same directory with the name [filename]_prediction.sa
This file is in the same format as the .sa files and it contains the given models predicted outputs.

RSAPredictionModel.py is the program that creates the RSAmodel.pkl file.
There is no need to run it as I have already done run it on my machine and its output, RSAmodel.pkl, is given

When RSAPredictionModel.py is run, it outputs the precision, recall, and F1 score for the model it outputs.
The precision, recall, and F1 scores it output for the model it produced, RSAmodel.py, are copy and pasted below.

Model Accuracy:
_______________
Precision: 0.6808739255014327
Recall: 0.731998459761263
F1: 0.705511226572648

 
 