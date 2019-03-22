//Hello
/*
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
WRITTEN BY OTHER STUDENTS - NAOMI KEUSCH BAKER AND MARTIN SCHREINER
*/
import java.util.*;
import java.io.*;
import java.lang.*;
import weka.core.converters.ConverterUtils.DataSource;

 import weka.core.Instances;
 import weka.experiment.InstanceQuery;

 InstanceQuery query = new InstanceQuery();
 query.setUsername("nobody");
 query.setPassword("");
 query.setQuery("select * from whatsoever");
 // You can declare that your data set is sparse
 // query.setSparseData(true);
 Instances data = query.retrieveInstances();
