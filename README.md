# WH-QG-QA
This repo contains the code for the WH QUestion Generation, Ranking, and Answering project for CMU's 11611 

# Question Ranking
There were three different types of scores as shown below
  Difficulty Score
  - Trained on human-labelled WH-Question difficulty dataset
  - Converted questions  to POS tags equivalent
  Trained a random forest regression model to predict difficulty from 1 (easy) - 3 (hard).

  Language Model Score
  - Used to model the syntactic correctness of a question 
  - Trained on the SQuAD dataset
  - Converted questions to POS tags equivalent
  - Trained a trigram language model on the POS tags
  - Score was perplexity score obtained using the model

  Similarity Score
  - Used to drop duplicate/similar questions
  - Based on the cosine similarity of the tf-idf vectors of the questions to being compared


# Question Generation 
  
