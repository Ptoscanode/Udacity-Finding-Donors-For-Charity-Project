# Udacity-Finding-Donors-For-CharityML


### Project Motivation
This projects consists of applying supervised learning techniques and an analytical mind on data collected for the U.S. census to help CharityML (a fictitious charity organization) identify the people most likely to donate to their cause. 

The project can be broken down into the following steps:

1 - Explore the data to learn how the census data is recorded
<br/>
2 - Apply a series of transformations and preprocessing techniques to manipulate the data into a workable format
<br/>
3 - Evaluate several supervised learners and consider which is best suited for the solution
<br/>
4 - Optimize the selected model and present it as my solution to CharityML.
<br/>
5 - Explore the chosen model and its predictions under the hood, to see just how well it's performing when considering the data it is given.


### Requirements

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)


### Files in the Repository 

`finding_donors.ipynb`: Jupyter notebook containing the analysis of the project

 `visuals.py`:   
 
 `census.csv`: The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

`feature_importance.png`: Bar chart with the five most important predictive features according to the model


### Results
Based on training/predicting time, F-score, data size and how cleansed the data is, the model to be chosen was **Gradient Boosting**.

Then, after tuning the model using GridSearch, we have the following numbers:

|     Metric     | Unoptimized Model | Optimized Model | Naive Predictor
| :------------: | :---------------: | :-------------: | :-------------:
| Accuracy Score |      0.8630       |      0.8700     |     0.2478
| F-score        |      0.7395       |      0.7486     |     0.2483


## Feature Importance

![Image](https://github.com/Ptoscanode/Udacity-Finding-Donors-For-CharityML/blob/main/feature_importance.png)

The features identified by the GradientBoost algorithm and the ones I believed would be the most important are not the same.

**My features:**

- Occupation

- Hours-per-week

- Education level

- **Capital gain**

- **Age**


**GradientBoost's features:**

- **Capital gain**

- Capital loss

- MaritalStatus_married-civ-spouse

- **Age**

- Education-num
 
**Capital gain** and **Age** matched, whereas Occupation and Education level didn't

To be honest, the fact that **Capital-loss**, **Marital Status** and **Education-num** are not listed is a surprise. 

I believed that **capital gain** would be more relevant than **capital loss** and that **Education level** would be more relevant than **Education num**. Maybe people who lost money are more sensitive to donate than those who have a high capital gain because they know how it is like. When it comes to Education-num, I really can't make an assumption.


## Feature Selection 

When we used only a subset of all the available features in the data both the reduced model's F-score and accuracy are lower than those of the model including all features. 

Accuracy went down from **0.8700** to **0.8579** and the F-Score from **0.7486** to **0.7225** 

If it was a real time application and if time was a constraint, using the reduced model could be helpful in providing some useful insights.

However, because the F-score has decreased from **0.7486** to **0.7225** , I would recommend to use the model built based on the entire dataset.

Unless time is a constraint and you need to deliver results as soon as possible. In the end, it boils down to a trade-off between time to deliver your results and predicting with more accuracy and precision whether a candidate makes more than $50,000 or not.


### Acknowledgements

**Random Forests**

https://en.wikipedia.org/wiki/Random_forest

https://www.mygreatlearning.com/blog/random-forest-algorithm/
<br/>

**Logistic Regression**

https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/

https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression
<br/>

**AdaBoost**

https://blog.paperspace.com/adaboost-optimizer/
<br/>


**Support Vector Machine**

https://dhirajkumarblog.medium.com/top-4-advantages-and-disadvantages-of-support-vector-machine-or-svm-a3c06a2b107

https://www.kdnuggets.com/2017/02/yhat-support-vector-machine.html#:~:text=SVM%20is%20a%20supervised%20machine,boundary%20between%20the%20possible%20outputs.
<br/>


**Gradient Boosting**

https://en.wikipedia.org/wiki/Gradient_boosting

https://corporatefinanceinstitute.com/resources/knowledge/other/boosting/

https://medium.com/gradient-boosting-working-limitations-time/gradient-boosting-working-and-applications-28e8d4ba866d
<br/>


**Bagging**

https://en.wikipedia.org/wiki/Bootstrap_aggregating

https://corporatefinanceinstitute.com/resources/knowledge/other/bagging-bootstrap-aggregation/
