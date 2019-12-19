# Machine Learning EPFL 2019 - Project 2

## TEPFL ML Text Classification 2019

### Abstract :

Text  classification  is  a  challenging  problem  thathas  many  applications  such  as  spam  filtering,  language  iden-tification...   Researchers   started   to   address   the   problem   inthe  1980s  but  with  the  recent  explosion  of  data  availableand  advances  in  Machine  Learning,  the  field  gained  a  lot  oftraction. This report details our work on the ”EPFL ML TextClassification  Challenge”  where  the  goal  is  to  use  machinelearning  to  classify  tweets.  We  have  to  predict  if  the  tweetcontained  a  ”:)”  or  a  ”:(”  smiley.  Thus  we  need  to  classify  atweet as positive or negative. 

### Introduction :

Hosted on the competitive platform AIcrowd, this projectaims to introduce us to text classification tasks. Competitorsare evaluated using the accuracy and the F1- score of theirpredictions.The training dataset contains 2,458,297 tweets. 1,218,655of them are labelled as positive (they contained a ”:)” smiley)and 1,239,642 are labelled as negative (they contained a ”:(”smiley). In the report, we will first explore the dataset. We will thentry  several  approaches:  computing  tweet  embedding  fromwords  embedding  and  training  a  classifier  model  using  theobtained vectors, using a neural network called fasttext, andusing transformer model. Finally we will present our resultsand discuss our work

### Directory structure

The following directory contrains different text documents, code and data files. The structure is detailed below :

#### Documents

- `project2_description.pdf`: Describe the task to perform and the tools availables.
- `project2_report.pdf`: Describe our approach, work, and conclusion while solving the problem.

#### Code

##### Jupyter Notebooks
- [Model Exploration](./scripts/model_exploration.ipynb): Test different machine learning models on the data to select the most performant.
- [Parameters Selection](./scripts/params_selection.ipynb): Use cross validation on the most performant model to define the best data processing and regularisation parameter.
- [Final Model](./scripts/final_model.ipynb): Use the best model and parameters to achieve the best accuracy and create a submission.

##### Python scripts

- [Implementations.py](./scripts/implementations.py): All possible machine learning models available to use.
- [Run.py](./scripts/run.py): Optimal data preprocessing and model to solve the problem.


- [Process Data](./scripts/process_data.py): All the preprocessing methods and tools used.
- [Proj1 Helpers](./scripts/proj1_helpers.py): General helpers methods for file submission, prediction, trianing and data splitting.
- [Least squares helpers](./scripts/least_squares_helpers.py): Helpers methods for least squares models (costs, optimisation).
- [Logistic regression helpers](./scripts/logistic_regression_helpers.py): Helpers methods for logistic regression (costs, optimisation).
