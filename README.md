# WH-QG-QA
This repo contains the code for the WH QUestion Generation, Ranking, and Answering project for CMU's 11611 - Natural Language Processing

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
- Used an ensemble of rule-based approach, information extraction techniques, and transformer-based architecture to generate questions
- Identify potential sentences for question generation targeting specific named entities such as PER, LOC, FAC, etc.
- Remove similar questions and rank them using the ranking module.

# Question Answering
- Get article and question embeddings using BERT-based architectures
- Find context sentence for answers using fuzzy matching and semantic search
- Pass context and question into a finetuned T5 based model and obtain answer

# Docker image
docker pull dehbaiyor/oaadebay_whqgqa:2704b

# References
- https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap
- https://github.com/mmxgn/spacy-clausie
- https://huggingface.co/transformers/custom_datasets.html#qa-squad
- https://www.kaggle.com/rtatman/questionanswer-dataset
