# Betamore Presents Python for Data Science


With the advent of Machine Learning and Big Data, data scientists who can create large-scale data-driven solutions are in high demand. By taking this four week compressed lesson, you will be able to utilize the python programming language in order to analyze both small and large data sets while building and evaluating your very own machine learning models.

**About the Instructor:**

![alt tag](images/sinan-headshot.jpg)

*Sinan Ozdemir* 

is a lecturer of Business, Mathematics and Computer Science at The Johns Hopkins University. Sinan is also a co-Founder/CTO of Tier5 and Kovani, two data science companies based in Baltimore.
Sinan is an experienced teacher and entrepreneur.
[Follow him!](https://twitter.com/intent/user?screen_name=prof_oz)

 

Day | Part | Topic
--- | --- | ---
4/15 | 1 | [Data Exploration with Pandas](#class-1-introduction-and-pandas)
4/22 | 2 | [Intro to Machine Learning](#class-2-introduction-to-machine-learning)
4/29 | 3 | [Model Evaluation and Metrics](#class-3-model-evaluation-metrics-and-procedures)
5/6 | 4 | [Building a Model using Titanic Survival Data](#class-4-titanic-data-set)
 


***Required Pre-Reqs for ANY of the Parts:***

* Download the [Anacondas Distribution of Python](http://continuum.io/downloads)
* Prepare to learn the glory that is Data Science!
* Anyone who is not feeling up on their python or coding skills in general should check out [this resource](http://dataquest.io/missions) to practice



# Class 1: Introduction and Pandas

In this session our objective is to learn the basics of python and how to use python's pandas to explore data sets.

At the end of the Class Students Will Be Able To: 

* Use a module in python called Pandas in order to explore small and large data sets
* Prepare data and prefrom necessary pre-processing steps

**Recommended Prereqs:**

* Ability to write and read code
* Background in Python/R is preferred but not required


Agenda:

* Introductions
* Intro to Data Science 
	* [slides](slides/01_intro_to_data_science.pdf)
* Introduction to Python with Pandas 
	* [code](code/01_pandas.py)

Homework

* [Practice Problems for next time!] (homework/01_pandas_homework.py) 
	* The [Data](data/drinks.csv) was taken from this [538 article](http://fivethirtyeight.com/datalab/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/)

	# Class 2: Introduction to Machine Learning

In this session we will go over the interesting subject of machine learning and build our first two models using the python package sci-kit learn.

At the end of the Class Students Will Be Able To: 

* Understand at a fundamental level what machine learning is and how it is used in practice.
* Use a module in python called Sci-kit learn in order to build and evaluate machine learning models
* Understand key differences between machine lerning models

**Recommended Prereqs:**

* Basic understanding of the Python package Pandas
* Background in Python/R is preferred but not required


**Agenda:**

* Iris dataset
    * [What does an iris look like?](http://sebastianraschka.com/Images/2014_python_lda/iris_petal_sepal.png)
    * [Data](http://archive.ics.uci.edu/ml/datasets/Iris) hosted by the UCI Machine Learning Repository
* Machine learning and KNN ([slides](slides/02_intro_to_machine_learning_knn.pdf))
    * [Reddit AMA with Yann LeCun](http://www.reddit.com/r/MachineLearning/comments/25lnbt/ama_yann_lecun)
    * [Characteristics of your zip code](http://www.esri.com/landing-pages/tapestry/)
    * KNN [code](code/02_knn.py)
* Introduction to Linear Regression
	* 	[slides](slides/02_linear_regression.pdf)
	* 	[code](code/02_linear_regression.py)

**Further Resources:**

* To go much more in-depth on linear regression, read Chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), watch the [related videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) or read a [quick reference guide](http://www.dataschool.io/applying-and-interpreting-linear-regression/) to the key points in that chapter.
* This [introduction to linear regression](http://people.duke.edu/~rnau/regintro.htm) is much more detailed and mathematically thorough, and includes lots of good advice.
* This is a relatively quick post on the [assumptions of linear regression](http://pareonline.net/getvn.asp?n=2&v=8).

* Documentation: [user guide](http://scikit-learn.org/stable/modules/neighbors.html), [module reference](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors), [class documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

Homework

* [Practice Problems for next time!] (homework/02_glass_knn.py) 
* There is also a homework question in the linear regression code file to work on!

	# Class 3: Model Evaluation Metrics and Procedures
	
	We will discover the process and quantifiable metrics that we use to evaluate our machine learning models

At the end of the Class Students Will Be Able To: 

*  see how data scientists prepare and alter their models in order to maximize accuracy.


**Recommended Prereqs:**

* Basic understanding of Regression and Classification

**Agenda:**

* Model Evaluation
    * Go over basic [Procedure] (slides/03_model_evaluation_procedures.pdf)
    * look at different [Metrics](slides/03_model_evaluation_metrics.pdf)
    * [Code](code/03_model_evaluation.py)

    
**Further Resources:**

* Great video of [ROC/AUC curves](https://www.youtube.com/watch?v=OAl6eAyP-yo)

	# Class 4: Titanic Data Set
	
	Congratulations! You have made it this far :) Today we will be looking at the titanic data set on Kaggle.com to get a model that will tell us whether or not a person died on the Titanic.
	
	* The competition lives [here](https://www.kaggle.com/c/titanic)
		* Find the code [here](code/04_kaggle_titanic.py)
		* Our data lives in two files. the [in sample data](data/titanic.csv) and our [out of sample data](data/titanic_test.csv) are separate.
