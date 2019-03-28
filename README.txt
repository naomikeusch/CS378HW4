To run this program, you will need to adjust the command lines according to where the files are stored.
Compiling requires being in the correct directory and having the correct class path to weka.jar.
Running additionally requires adjusting the second directory (after the semi-colon) to where final_classifiers.java is located:
Note: this program prints a lot, but the run time is less than a minute. Feel free to comment out print statements as needed. Thank you.

javac -cp "C:\Program Files\Weka-3-8\weka.jar" final_classifiers.java
java -cp "C:\Program Files\Weka-3-8\weka.jar;C:\Users\naomi\...\hw4" final_classifiers
