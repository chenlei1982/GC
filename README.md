In this respository, codes form analyzing transcriptomic data of tumor microenvironment cells from gastric cancer patients are provided. These codes implement seven featrue ranking algorithms: last absolute shrinkage and selection operator (LASSO), light gradient boosting machine (LightGBM), Monte Carlo feature selection (MCFS), Minimum Redundancy Maximum Relevance (mRMR), random forest (RF), Categorical Boosting (CATBoost) eXtreme Gradient Boosting (XGBoost), and four classification algiorithms: decision tree (DT), k-nearest neighbors (KNN), random forest (RF), and support vector machine (SVM).

Python environment installation

Step 1 Download the software by git clone https://github.com/*******/*******

Step 2 Installation has been tested in Linux and Windows with Python 3 (Recommend to use Python 3.10 or miniconda 3 platform). Since the package is written in Python 3, Python 3 with the pip tool must be installed first. The tools use the following dependencies: numpy, scipy, pandas, scikit-learn You can install these packages first, by the following commands: pip install pandas pip install numpy pip install scipy pip install scikit-learn

Step 3 cd to the software folder and run the installation command: python *************

Guide for codes 

Step 1: The transcriptomic data of tumor microenvironment cells from gastric cancer patients was fed into the codes of one of the feature ranking algorithm, which can produce a feature list. 

Step 2: With the feature list in Step 1, using codes in folder csv_make to generate csv files, which contain samples represented by some top features in the list. 

Step 3: For each generated csv file, using the codes of one classification algorithm to obtain the cross-validation results.

Other notes

In each folder, the file "comd" contains the codes to run the corresponding package.

The input file of MCFS should be in adx format, whereas those for other algorithms are in csv format.
