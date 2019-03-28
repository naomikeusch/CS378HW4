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
import weka.classifiers.meta.Vote;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.Classifier;

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
 
 
//https://www.codingame.com/playgrounds/5844/machine-learning-with-java---part-4-decision-tree
 
public class ensemble {

  public static void main(String[] args) throws Exception {
	  
		// PrintWriter out = new PrintWriter("results.txt");
		
		String s = "";
		for(int i = 12; i<=23; i++){
			File data = new File("grades2.csv");
			CSVLoader loader = new CSVLoader();
			loader.setSource(data);
			Instances inst = loader.getDataSet();
			Remove remove = new Remove();
			s += i +", ";
			remove.setAttributeIndices(s + "last");
			remove.setInvertSelection(true);
			remove.setInputFormat(inst);
			Instances train = Filter.useFilter(inst, remove);
		  
			train.setClassIndex(train.numAttributes() - 1);
			
			Evaluation eval_mlp = new Evaluation(train);

		  
			MultilayerPerceptron mlp = new MultilayerPerceptron();
	 
			mlp.buildClassifier(train);

			
			eval_mlp.crossValidateModel(mlp, train, 10 , new Random(1));
			
			System.out.println("MLP - Quizzes up to " + (i-11));
			System.out.println(eval_mlp.toSummaryString("\nResults\n=====\n", true));
			System.out.println("Accuracy: " + eval_mlp.toClassDetailsString());
			System.out.println(eval_mlp.toMatrixString());
			
			Classifier tree = new RandomTree();
			
			tree.buildClassifier(train);
			
			Evaluation eval_tree = new Evaluation(train);
			
			eval_tree.crossValidateModel(tree, train, 10 , new Random(1));
			
			System.out.println("Tree - Quizzes up to " + (i-11));
			System.out.println(eval_tree.toSummaryString("\nResults\n=====\n", true));
			System.out.println("Accuracy: " + eval_tree.toClassDetailsString());
			System.out.println(eval_tree.toMatrixString());
		
		  
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(train);
			
			Evaluation eval_nb = new Evaluation(train);
			
			eval_nb.crossValidateModel(nb, train, 10 , new Random(1));
			
			System.out.println("Naive Bayes - Quizzes up to " + (i-11));
			System.out.println(eval_nb.toSummaryString("\nResults\n=====\n", true));
			System.out.println("Accuracy: " + eval_nb.toClassDetailsString());
			System.out.println(eval_nb.toMatrixString());
			
			
			Vote vote = new Vote();
			vote.addPreBuiltClassifier(nb);
			vote.addPreBuiltClassifier(mlp);
			vote.addPreBuiltClassifier(tree);
			vote.buildClassifier(train);
			
			
			
			Evaluation eval_vt = new Evaluation(train);
			
			
			eval_vt.crossValidateModel(vote, train, 10 , new Random(1));
			System.out.println("Ensemble - Quizzes up to " + (i-11));
			System.out.println(eval_vt.toSummaryString("\nResults\n=====\n", true));
			System.out.println("Accuracy: " + eval_vt.toClassDetailsString());
			System.out.println(eval_vt.toMatrixString());
		}
  }
}