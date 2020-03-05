## Dependencies
Python3 and some python3 libraries:
 - numpy (v1.15.4 used)
 - scipy (v1.1.0 used)
 - sklearn (v0.20.0 used)
 - imblearn (v0.4.3 used) https://github.com/scikit-learn-contrib/imbalanced-learn
 - matplotlib (to plot, v3.0.3 used)


## Content
folder 'datasets':
 - contain the 22 datasets used in the experiments

folder 'experiments_a_b':
 - contain the source code to reproduce experiments a and b from the paper:
   without data processing (a) and with SMOTE and Random Undersampling (b)
 - the experiment is launched with 'main.py' which apply the algorithms on the
   datasets, and store the results in a subfolder
 - the file 'latex.py' load the result files, perform a mean over the files and
   generate latex code producing an array of the mean results

folder 'experiments_c_d':
 - contain the source code to reproduce experiments c and d from the paper
   where we artificially increase the imbalance in the datasets
 - the experiment is launched with 'main.py' which apply the algorithms on the
   datasets, and store the results in a subfolder
 - the files 'plotAccuracyF1.py' and 'plotF1.py' load the results files,
   perform a mean over the files and generate PDF images of the results
