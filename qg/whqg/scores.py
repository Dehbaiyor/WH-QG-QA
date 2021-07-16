import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pandas as pd
    import numpy as np
    import dill as pickle

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from joblib import load
    import pandas as pd
    import nltk
    
def return_tag(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return " ".join([tag[1] for tag in tagged])

clf = load('./qg/models/rf.joblib') 
vect = load('./qg/models/vect.joblib') 

with open("./qg/models/vocab.sv", 'rb') as in_strm: 
    vocab = pickle.load(in_strm)
    
def replace_rare_tags(text):
    text = text.split(" ")
    for i in range(len(text)):
        word = text[i]
        if word not in vocab:
            text[i] = "oov"
    return " ".join(text)

def get_difficulty_score(text):
    return clf.predict(pd.DataFrame.sparse.from_spmatrix(vect.transform([return_tag(text)])))

def get_sim_score(questions):
    df = pd.DataFrame(list(zip(range(len(questions)),questions)), columns = ["ID","Qsts"])
    corpus = list(df["Qsts"].values)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(X, X)
    return np.mean(sims, axis = 1)

with open("./qg/models/model.sv", 'rb') as in_strm:
    model = pickle.load(in_strm)  
    
def get_lm_score(text):
    text = replace_rare_tags(return_tag(text))
    text = text.split(" ")
    score = 0
    for i in range(len(text)):
        if i == 0:
            s = model[(None,None)][text[0]]
            if s == 1:
                s = 0.000000001
            score += np.log(s)
        elif i == 1:
            s = model[(None,text[0])][text[1]]
            if s == 1:
                s = 0.000000001
            score += np.log(s)
        else:
            s = model[(text[0],text[1])][text[2]]
            if s == 2:
                s = 0.000000001
            score += np.log(s)
    score = score/len(text)
    return 2**-score