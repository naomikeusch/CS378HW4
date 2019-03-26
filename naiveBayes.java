/*
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
WRITTEN BY OTHER STUDENTS - NAOMI KEUSCH BAKER AND MARTIN SCHREINER
*/

import java.util.*;
import java.io.*;
import java.lang.*;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

//Command prompts:
//javac -cp "C:\Program Files\Weka-3-8\weka.jar" classifiers.java
//java -cp "C:\Program Files\Weka-3-8\weka.jar;C:\Users\naomi\OneDrive\Documents\Emory\Third Year\cs 378\hw4\hw4" classifiers

//References:

/**
 * This example trains NaiveBayes incrementally on data obtained
 * from the ArffLoader.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
 
 // https://www.youtube.com/watch?v=q3Gf6kqaJWA
 
public class naiveBayes {

  public static void main(String[] args) throws Exception {
	  
		File data = new File("grades2.csv");
	    CSVLoader loader = new CSVLoader();
        loader.setSource(data);
        Instances inst = loader.getDataSet();
		Remove remove = new Remove();
		remove.setAttributeIndices("12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, last");
		remove.setInvertSelection(true);
        remove.setInputFormat(inst);
		Instances train = Filter.useFilter(inst, remove);
	  
		train.setClassIndex(train.numAttributes() - 1);

	  
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
	  
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(nb, train, 10 , new Random(1));
		System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
		System.out.println("Accuracy: " + eval.toClassDetailsString());
		System.out.println(eval.toMatrixString());
  }
}