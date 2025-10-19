Kaiburr Task 5 — Consumer Complaint Text Classification



This repository contains my submission for Kaiburr Assessment 2025 (Task 5).

It performs text classification on the Consumer Complaint dataset from the US CFPB portal.

The model classifies each complaint into one of four categories:



0 – Credit reporting, repair, or other



1 – Debt collection



2 – Consumer Loan



3 – Mortgage





How to Run



1\. Clone this repository and open it in your terminal:



git clone https://github.com/ratnakumarmallisetty/kaiburr-task5.git

cd kaiburr-task5



2\. Create and activate a virtual environment:



python -m venv .venv

.venv\\Scripts\\activate



3\. Install dependencies:



pip install -r requirements.txt



4\. Place the complaints.csv file inside the data/ folder.



5\. Run the pipeline:



python src/prepare.py

python src/train.py

python src/predict.py --text "They keep calling about an old debt I do not owe."



Results Summary:


----------------------------------------------------|
Model	              Accuracy	 Macro F1   Selected  |
----------------------------------------------------
Logistic Regression	0.89	  0.65	      Best        |
----------------------------------------------------
Linear SVM	        0.92	  0.63	       –          |
-----------------------------------------------------




Example prediction:

\[predict] 1 :: Debt collection :: They keep calling about an old debt I do not owe.





Artifacts (can be accessed in the src folder):



artifacts/train\_report.txt — evaluation metrics



artifacts/best\_model.joblib — serialized model



artifacts/X\_y\_summary.json — dataset summary



Tools Used:



Python 3.12



pandas, scikit-learn, joblib



TF-IDF + Logistic Regression / Linear SVM



Screenshots:

Outputs of prepare.py, train.py and prepare.py are there in the screenshots folder.





Author:

Mallisetty Rathna kumar

B.Tech Computer Science \& Engineering — Amrita Vishwa Vidyapeetham

(Kaiburr Assessment 2025 Candidate)



