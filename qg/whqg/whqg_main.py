import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from qg.whqg.whqg_helper import *
    from qg.whqg.scores import *
    #from transformers import pipeline
    #summarizer = pipeline("summarization")

def strip_newline(txt):
    return txt.replace("\n","")

def drop_duplicate_questions(questions):
    df = pd.DataFrame(list(zip(range(len(questions)), questions)), columns = ["ID","Qsts"])
    corpus = list(df["Qsts"].values)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    q_wdup = [questions[i] for i in list(pd.DataFrame.sparse.from_spmatrix(X).drop_duplicates().index)]
    return q_wdup

#def summarize(sentence):
#    length = len(sentence.split())
#    return summarizer(sentence, max_length = int(0.95 * length), min_length = int(0.8 * length), do_sample=False)[0]['summary_text']


def get_wh_questions(filepath, n):
    f = open(filepath, encoding = "utf-8", mode = "r")
    txt = f.read()
    txt = txt[txt.find("\n\n\n")+3:txt.find("References")]
    
    qs_ents = []
    qs_trans = []
    qs_ie = []
    
    #para = ""
    #paragraphs = txt.split("\n\n")
    #for paragraph in paragraphs:
    #    if len(paragraph) > 20:
    #        para += summarize(paragraph)
    #        
    #txt = para
    
    questions_ent, sentences, ents1, ents2, ents3 = generate_questions_ent(txt)
    
    gen = pd.DataFrame(questions_ent)
    gen['lm_score'] = gen[0].apply(get_lm_score)
    #gen['sim_score'] = get_sim_score(questions_ent)
    gen['difficulty_score'] = gen[0].apply(get_difficulty_score)
    gen['lm_score'] = gen['lm_score'].rank(pct = True,  ascending = False)
    #gen['sim_score'] = gen['sim_score'].rank(pct = True, ascending = False)
    #gen['difficulty_score'] = gen['difficulty_score'].rank(pct = True, ascending = False)
    gen['difficulty_score'] = gen['difficulty_score']/gen['difficulty_score'].max()
    gen['rank_score'] = gen['lm_score'] * gen['difficulty_score']
    sorted_qs = gen.sort_values(by = 'rank_score', ascending = False)
    qs_ents = list(sorted_qs[0].values)
    qs_ents =  drop_duplicate_questions(qs_ents)
    
  
    questions_t = generate_questions_transformer(sentences, n)
  
    gen = pd.DataFrame(questions_t)
    gen['lm_score'] = gen[0].apply(get_lm_score)
    #gen['sim_score'] = get_sim_score(questions_t)
    gen['difficulty_score'] = gen[0].apply(get_difficulty_score)
    gen['lm_score'] = gen['lm_score'].rank(pct = True,  ascending = False)
    #gen['sim_score'] = gen['sim_score'].rank(pct = True, ascending = False)
    #gen['difficulty_score'] = gen['difficulty_score'].rank(pct = True, ascending = False)
    gen['difficulty_score'] = gen['difficulty_score']/gen['difficulty_score'].max()
    gen['rank_score'] = gen['lm_score'] * gen['difficulty_score']
    sorted_qs = gen.sort_values(by = 'rank_score', ascending = False)
    qs_trans = list(sorted_qs[:n][0].values)
    qs_trans =  drop_duplicate_questions(qs_trans)
    
 
    questions_ie = return_questions_ie(sentences, ents1, ents2, ents3)
    
    gen = pd.DataFrame(questions_ie)
    gen['lm_score'] = gen[0].apply(get_lm_score)
    #gen['sim_score'] = get_sim_score(questions_ie)
    gen['difficulty_score'] = gen[0].apply(get_difficulty_score)
    gen['lm_score'] = gen['lm_score'].rank(pct = True,  ascending = False)
    #gen['sim_score'] = gen['sim_score'].rank(pct = True, ascending = False)
    #gen['difficulty_score'] = gen['difficulty_score'].rank(pct = True, ascending = False)
    gen['difficulty_score'] = gen['difficulty_score']/gen['difficulty_score'].max()
    gen['rank_score'] = gen['difficulty_score'] * gen['lm_score']
    sorted_qs = gen.sort_values(by = 'rank_score', ascending = False)
    qs_ie = list(sorted_qs[0].values)
    
    questions =  drop_duplicate_questions(qs_ie + qs_trans)
    
    gen = pd.DataFrame(questions)
    gen['lm_score'] = gen[0].apply(get_lm_score)
    #gen['sim_score'] = get_sim_score(questions)
    gen['difficulty_score'] = gen[0].apply(get_difficulty_score)
    gen['lm_score'] = gen['lm_score'].rank(pct = True,  ascending = False)
    #gen['sim_score'] = gen['sim_score'].rank(pct = True, ascending = False)
    #gen['difficulty_score'] = gen['difficulty_score'].rank(pct = True, ascending = False)
    gen['difficulty_score'] = gen['difficulty_score']/gen['difficulty_score'].max()
    gen['rank_score'] = gen['lm_score'] * gen['difficulty_score']
    sorted_qs = gen.sort_values(by = 'rank_score', ascending = False)
    
    questions = list(sorted_qs[:n][0].apply(strip_newline).values)
    if len(questions) < n:
        questions = questions + qs_ents[:n - len(questions)]
    
    return questions
    