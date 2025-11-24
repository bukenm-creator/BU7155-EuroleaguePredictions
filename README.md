# BU7155-EuroleaguePredictions

This code uses Euroleague history data and tries to predict 2025-2026 season matches by using GMMM algorithm and then creates some betslips according to the low-moderate-high risk levels by using tree search algorithms. Risk levels are given below.

* Low Risk Level: Combination of 3 matches with combined probability over 55%
* Moderate Risk Level: Combination of 3 matches with combined probability between 25%-55%
* High Risk Level: Combination of 3 matches with combined probability between 10%-25%

It gives the Accuracy, Precision, RMSE, ME, AUC, MAPE scores of the system by applying model to train and validation datasets. All data is taken from Euroleague's own api by using following code repo on GitHub. 

You can use added CSV and Excel Files "Euroleague_Schedule", to see create predictions file and betslips file. Those output files are also added, if you just wanna check whether 

* euroleague-box_score: Detailed match statistics such as FG/2PT/3PT (Attempts and Success, Total Rebounds, Fouls, Steals etc.
* euroleague-header: General match details such as referee name, home/away team score for each quarter etc.
* Euroleague_Schedule: 2025-2026 Season's euroleague schedule
* Euroleague_Predictions: Output files that uses euroleague-header and euroleague-box-score files and create current season's match results which are written in Euroleague_Schedule file. It gives probability of both team's winning and creates some odds accordingly for each matches
* betslips: 3 Matches combination for all risk levels and rounds, it gives combined odds for each option.



https://github.com/bsamot10/EuroleagueDataETL/tree/main
