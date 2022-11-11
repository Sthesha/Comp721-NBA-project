# Comp721-NBA-project
This is the machine learning project that deals with prediction of game out comes and players who are outliers in the NBA game. 

COMP721- Machine Learning
Project Report
The National Basketball Association (NBA) Prediction
Kamlin Pillay– 217047298
Luthando Nxumalo- 220111032
Merwyn Moodly- 219095202
Siphiwe Maphalala- 217034316
Sthembiso Mkhwanazi- 217058845
Abstract
The NBA is the premier basketball league in the world. The growth of the sports betting industry as
well as the increasing integration of machine learning in sports science and statistical analysis has
highlighted the need for research in the areas of outlier detection and outcome prediction. This
project addresses these tasks with several machine learning approaches. Outstanding player
detection was performed by 4 classifiers, with the AdaBoost classifier outperforming others with an
accuracy of 86.54%. For game outcome prediction, 2 machine learning approaches were explored, a
Logistic Regression model and a Random Forest model. The Logistic Regression model performed
most favourably with an accuracy of 92.4%.
1. Introduction
The National Basketball Association1
(NBA) is a professional basketball league based in North
America. The league is comprised of thirty teams and is ranks highly amongst major sports leagues in
Canada and USA in terms of annual revenue. The NBA is also recognised as the best basketball
league, globally [6].
A regular NBA season runs from October to April, with every team playing 82 games. The leagues
playoff tournament extends into June. When a game is played, the possible outcomes for a team are
win, lose, or draw.
The term outliers, or outstanding players, refer to those players that have stood out, statistically
speaking from other players in the league. These include exceptionally high scoring players, players
that score more points of a specific type (3-pointers2
, penalties etc.) but also players that perform
exceptionally in other aspects of the game such as transitional plays and defence.
Outstanding players are eligible for various awards and titles, such as being named the leagues most
valuable player (MVP). Due to the impact of these exceptional performers and their influence on
games, the ability to identify them through statistical analysis is in high demand. This demand comes
from various sources such as the fans, the sports betting industry, potential future teams, various
organisation offering sponsorship or potential brand ambassador roles, and strategic planners within
the sport, to name a few.
The continuously growing interest in outlier detection and game outcome prediction serves as
additional motivation for this project. The first part of this project deals with detecting the
outstanding players. This is done by considering relevant data and performing feature engineering.
Three classification models were used: AdaBoost, K-Nearest Neighbours and Decision Tree Classifier.
The performance of these models was then evaluated and analysed. The second part of this project
addresses the demand for game outcome prediction. First, a dataset was formed using a
combination of player and team statistics from the given dataset. This new dataset was then used to
create features that represent the strength of a team or conversely, the difficulty of a fixture against
the team. The new data was then used to train and test various prediction models. The
performances of these models were then evaluated and compared.
The remainder of this paper will be presented in the following sections: section 2 provides a critical
synthesis of the related literature, section 3 describes the methods and techniques employed for
1 https://www.nba.com/news/about
2 https://www.rookieroad.com/basketball/shot-types/3-pointer/
feature selection, model selection and metrics of evaluation, section 4 presents the results obtained
and discussion of these results, section 5 contains the concluding statements and possible
extensions/future work. Section 6 provides the references.
2. Related Work
The work done in [1] was closely related to the tasks addressed by this project. The study reports the
experimental results obtained from various machine learning approaches to outlier detection and
game outcome prediction in the NBA. Similarly, to this project, [1] was motivated by the growing
business and fanbase interest in game outcome prediction and MVP prediction. The dataset used
was comprised of 34 features and the began from the 1980 NBA season- when 3-pointers were
introduced. Various approaches for feature selection (BIC3
, PCA4
, and stepAIC5
) were applied along
with linear regression and SVM Regression models for team win ratio of per year prediction.
Outstanding players are predicted based on the various attributes such as rebounds, steals, three
points etc. Random Forest produced the best model of the study for outcome prediction, with an
accuracy of 61%. Outlier detection was approached with unsupervised learning techniques. The
models designed were able to detect 14 of 16 correct outliers (MVPs), from a pool of 30 outliers.
Multi-variate outlier detection based on chi-square distribution and k-means clustering gave
superior results compared to the study’s DBSCAN (Density-based spatial clustering of applications
with noise) approach. The study also concludes that in the case of unsupervised learning for outlier
detection, ground truth is not available. Hence there is no way to be empirically certain that the
designed algorithm produced the intended results.
Prediction of NBA games using machine learning techniques were performed in [2]. Three datasets
were created and used: player-only statistics, team-only statistics and team and player statistics.
Aside from testing various models on these datasets to observe how each dataset and model effect
the game outcome, [2] also focused on web scraping techniques and feature engineering on the
datasets to best suit the classification problem. In addition to standard features such as average wins
and points scored, emphasis was given to the home or visitor attribute as well as the previous 8
games of a team, which, when averaged, reflect the teams current form. Their Multilayer Perceptron
Classifier produced the best results with an accuracy of 66.8% and used a combination of team and
player statistics. It is also important to note that using player-only statistics produced less favourable
results than using team-only statistics. This indicates that game outcomes are more closely related
to team statistics rather than individual player statistics. A possible improvement to [2] would be to
use a weighted average for team data instead of the historical average, this way more weight can be
allocated to more recent team data.
The aim of work done in [3] was to build on existing machine learning techniques to produce a
better performing NBA game outcome predictor. The dataset used comprised of an array of team
statistics for both, home and away teams for each matchup. Two supporting features were feature
engineered. Six different models- Logistic Regression, Random Forest Classifier, K Neighbors
3 https://www.sciencedirect.com/topics/social-sciences/bayesian-information-criterion
4 https://www.simplilearn.com/tutorials/machine-learning-tutorial/principal-component-analysis
5 https://stats.stackexchange.com/questions/347652/default-stepaic-in-r
Classifier, Support Vector Classifier, Gaussian Naïve Bayes, and XGBoost Classifier- were trained and
tested and the best performing model was the Gaussian Naïve Bayes. An exhaustive grid search was
implemented for hyperparameter tuning to refine the model. The final model reported an accuracy
of 65.1%. A possible improvement to this work would be to use more live datasets, that are
concurrently updated to reflect the current performances of team, for more accurate prediction
results.
One of the most popular models used to predict NBA games is the “FiveThirtyEight” model found in
[4]. A special metric called Elo rating is used. Elo ratings monitor three key attributes: game score,
game location and the date-time details of the game. Elo ratings are set to 1500 as the default value
and this value changes depending on the teams’ performance throughout a season. The teams’ Elo
ratings are carried over from season to season because positive and negative trends in performance
carry over into new seasons. The model also uses a feature called RAPTOR. This feature is a
combination player tracking metrics and general player stats. Both features, Elo and RAPTOR, are
updated after each game played. The model also considers player fatigue, home and away
advantage, altitude based on game location and playoff complications. The model is very complex
and is the best performing NBA prediction model to date.
When researching the task of predicting game outcome, we came across a paper by Jasper Lin,
Logan Short, and Vishnu Sundaresan [7]. Lin and his associates from Stanford University used NBA
box scores from 1991-1998 to predict the winner of professional NBA games. Their initial test of
simply using the team with the higher win percentage resulted in an accuracy of 63.48%. They then
introduced 5 different supervised learning models, Logistic Regression, SVM, AdaBoost, Random
Forest, and Gaussian Naïve Bayes. Lin then used a dataset consisting of various NBA statistics
including points scored, field goals attempted etc. They used various benchmarks for evaluation of
the models. Other than the win percentage stated above, a team’s point differential (the difference
between the average points per game and average points allowed per game), and a win prediction
accuracy of experts of the field which is around 71%. The expert taken accuracy may seem
impressive, however, the model did not predict a winner on games that were deemed “too close to
call” thus inflating its accuracy. Other than logistic Regression which performed well, they realized
that the models suffered from overfitting and poor test accuracies. To correct this, multiple feature
selection algorithms were utilized to select only the features that impacted the accuracy the most.
The two feature selection algorithms used were forward and backward search, in which they used
10-fold cross validation to add or remove features one by one to determine which features result in
the highest prediction accuracies. Additionally, a heuristic feature selection algorithm was used to
verify the result of the other two algorithm. The team used backward search as the algorithm of
choice after tests were conducted. Recent performance records were used to test the hypothesis of
the Hot Hand Fallacy, which states that players and teams doing well will continue to do well,
initially believed to be false, utilizing a new approach may prove to agree with the claim after all. The
study concluded that after including 15 features, the accuracy was 2% lower than the initial accuracy
of 63.48% given by the win percentage. This observation is likely since win percentage inherently has
information about how the team has been doing (the teams form).
3. Methods & Techniques
Subsections 3.1. (Outlier detection) and 3.2. (Outcome prediction) are presented in a similar format:
explaining the data used, feature selection and the details of each classification/prediction model.
Subsection 3.3. describes the metrics of evaluation used.
3.1. Outlier Detection
3.1.1. Feature selection
A feature selection algorithm (KBest Selector) was used to identify categorical data that was most
relevant to the outlier detection task at hand. An 80:20 split was used for training and testing,
respectively.
3.1.2. Classification Models
First, an AdaBoost classifier was used. The second classifier was a Decision Tree classification model,
the third model used was a K- Nearest Neighbours classifier and the final model used was a Random
Forest classifier.
3.2. Outcome Prediction
3.2.1. Feature selection
The first step was to combine player and team data into a single data frame. Once this was done, any
columns deemed irrelevant to the outcome of a match and any invalid entries were dropped. The
second step was to create the Fixture Difficulty Rating (FDR). For each team, a combination of
favourable team and player stats were used to create a feature that essentially describes the quality
and relative skill of a team. From the perspective of an opposing team, FDR describes how difficult a
given fixture may be- a higher FDR indicates a stronger team, and a more difficult fixture for the
opposing team.
The next key feature used was the win likelihood. This feature is a combination of attributes relating
to team win ratios, player win ratios, coach stats and overall seasonal stats.
Forward selection was the feature selection approach employed for outcome prediction. The model
achieved its best accuracy with 10 parameters, the addition of parameters exceeding 10 decreases
overall accuracy of the model.
3.2.2. Prediction Models
Again, an 80:20 train and test split was adopted. A Logistic Regression model and a Random Forest
model were used. The accuracies of the models were improved by iterative hyperparameter tuning.
We recognise that evaluation of both models on the training set leads to overfitting. The problem of
overfitting was addressed using cross-validation. A wide range of values were evaluated for each
hyperparameter to narrow the search for parameters.
3.3. Evaluation Metrics
Precision, f1 score, accuracy and recall were used to evaluate the classifiers performance.
The precision value serves as an indication of the number of correct predictions (classifications)
made compared to incorrect prediction for a given class.
F1 score is the harmonic mean value between precision and recall.
Accuracy is an indication of correct predictions made in relation to all predictions made.
Recall serves as a measurement that indicates the number of correct classifications made for a class,
in relation to the number of items in that class.
A confusion matrix was also implemented to further conceptualize the above evaluation methods
and evaluation of the classifier and its variations (outlier detection only).
4. Results & Discussion
Subsections 4.1. and 4.2. present the results of the models used and a summary of the results for
outlier detection and outcome prediction respectively.
The software for the project can be found at: https://github.com/Sthesha/Comp721-NBA-project.git
4.1. Outlier Detection
4.1.1. AdaBoost Classifier
Confusion matrix for AdaBoost:
4.1.2. Decision Tree Classifier
Confusion matrix for Decision Tree Classifier:
4.1.3. K-Nearest Neighbours Classifier
Confusion matrix for K-Nearest Neighbours Classifier:
4.1.4. Random Forest Classifier
Confusion matrix for Random Forest Classifier:
Summary of Outlier Detection Results:
From the four models used for outlier detection, the AdaBoost model produced the highest
accuracy of 86.5%. Table 1 below illustrates the performances of the models, ordered by
accuracy.
Model Accuracy
AdaBoost 0.8654
Decision Tree 0.7739
KNN 0.8459
Random Forest 0.8354
Table 1: Comparison of Outlier Detection Classifiers
4.2. Outcome Prediction
4.2.1. Logistic Regression Model
Results for the LR model:
4.2.2. Random Forest Model
Results for RF model:
Summary of Outcome Prediction Results:
From the two models used for outcome prediction, the LR model produced the higher accuracy
of 92.4% and outperformed the Random Forrest model, which produced an accuracy of 72.3%.
Table 2 below illustrates the performances of both models.
Model Accuracy
Logistic Regression 0.9236
Random Forest 0.7226
Table 2: Comparison of Outcome Prediction models
5. Conclusions
The project was a success as we were able to create working classification and prediction
models for both tasks (outstanding player detection and outcome prediction). Of the 4
models used for outlier detection, the AdaBoost model outperformed other modules with
an accuracy of 86.54%.
For outcome prediction, the logistic regression model performed with an accuracy of 92.4%.
The decision tree classifier performed less favourably, with an accuracy of 72.3%.
Possible improvements to this project would be to utilise a more expansive feature set,
although computationally more expensive, it is possible that iterative reconfiguration of
feature sets and dimensionality reduction of these varying feature sets may lead to
improved performance of some models.
Another possible improvement may be to use more recent stats for each team. An average
performance feature that considers the last 5 or 10 games that a team has played, weighted
appropriately will lead to improved performance of the outcome prediction model.
For both tasks, future work should be directed at created ensemble classifiers and predictors
that fully utilise the available data (and additional data from the original source). These
ensemble models will produce improved performances.
6. References
[1] Aravind Anantha, Abinav Pothuganti, Chethan Thipperudrappa, Huy Tu, Outlier Detection
and Game Outcome Prediction of NBA Game, North Carolina State University, North
Carolina, USA.
[2] Rohan Agarwal BITS Pilani K. K. Birla Goa Campus Goa, India 403726. Prediction of NBA
Games Using Machine Learning. Susmit Wani, Swapnil Ahlawat, Wandan Tibrewal.
[3] Matthew Houde, Predicting the Outcome of NBA Games. Bryant University, Smithfield,
Rhode Island, April 2021.
[4] Silver, Nate. “2019-20 NBA Predictions.” FiveThirtyEight, FiveThirtyEight, 12 Feb. 2020,
projects.fivethirtyeight.com/2020-nba-predictions/.
[5] https://watchstadium.com/which-nba-statistics-actually-translate-to-wins-07-13-2019/ -
Chinmay Vaidya
[6] https://bleacherreport.com/articles/1291287-power-ranking-the-best-basketball-leagues-inthe-world-outside-of-the-nba
[7] Jasper Lin, Logan Short, and Vishnu Sundaresan. Predicting National Basketball Association
Winners. 2014.
