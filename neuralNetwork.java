/*
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
WRITTEN BY OTHER STUDENTS - NAOMI KEUSCH BAKER AND MARTIN SCHREINER
*/

//Sources: https://sefiks.com/2017/02/20/building-neural-networks-with-weka/

import java.util.*;
import java.io.*;
import java.lang.*;

import weka.classifiers.pmml.consumer.NeuralNetwork;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;

public class neuralNetwork {

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
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();
 
		mlp.buildClassifier(train);

		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(mlp, train, 10 , new Random(1));
		
		
	  System.out.println(eval.toSummaryString("\nResults\n=====\n", true));
	  System.out.println("Accuracy: " + eval.toClassDetailsString());
	  System.out.println(eval.toMatrixString());
  
  }
   
}
