from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch
from fuzzywuzzy import fuzz
import pandas as pd
import spacy
#import claucy
from sklearn.metrics.pairwise import cosine_similarity as sim

tokenizer = DistilBertTokenizerFast.from_pretrained('./qa/finetuned_qa_model_combined/')
loaded_model = DistilBertForQuestionAnswering.from_pretrained('./qa/finetuned_qa_model_combined/')
bi_encoder = SentenceTransformer('./qa/nq-distilbert-base-v1', device = 'cpu')
nlp = spacy.load('en_core_web_lg')
#claucy.add_to_pipe(nlp)

def get_position(predicted_answers, actual_ans):
    max_ans = ""
    max_score = 95
    for ans in predicted_answers:
        match_score =  fuzz.token_set_ratio(ans.split(), actual_ans.split())
        #print(match_score)
        if match_score >= 99:
            return predicted_answers.index(ans) + 1
        if match_score > max_score:
            max_ans = ans
            max_score = match_score
    if max_ans == "":
        return 0
    return predicted_answers.index(max_ans) + 1

def get_triples(sentence):
    breakdown = []
    doc = nlp3(sentence) 
    breakdown = doc._.clauses
    return breakdown

def get_ans_ie(context, question):
    ans = []
    c_triples = get_triples(context)
    q_triples = get_triples(question)
    if len(q_triples) == 0:
        return ""
    for c_triple in c_triples:
       for q_triple in q_triples:
            if str(c_triple.verb).lower() == str(q_triple.verb).lower().split():
                ans.append(c_triple.subject)
    return ans

def fuzzy_score(text, question):
    return fuzz.token_set_ratio(text.split(), question.split())

def get_context(query, wiki_embeddings, passages, top_k = 3):
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, wiki_embeddings, top_k=top_k)
    hits = hits[0]
    context_sentences = []
    for hit in hits:
        context_sentences.append(passages[hit['corpus_id']][1])
    return context_sentences

def get_answer(question, text):
    inputs = tokenizer(question, text, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])#

    outputs = loaded_model(**inputs, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    answer_start = torch.argmax(start_scores) 
    answer_end = torch.argmax(end_scores) + 1

    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

def get_embs(filepath):
    f = open(filepath, encoding = "utf-8", mode = "r")
    txt = f.read()
    txt = txt[txt.find("\n\n\n")+3:txt.find("References")]
    sentences = sent_tokenize(txt)
    _sent = enumerate(sent_tokenize(txt))
    passages = [[str(index), sentence] for index, sentence in _sent]
    passage_embedding = bi_encoder.encode(passages)
    return passage_embedding, passages, sentences
    
def get_wh_answer(sentences, question, passage_embedding, passages, n_answers = 1):
    answers = []
    s = pd.DataFrame(sentences)
    s["1"] = s[0].apply(fuzzy_score, question = question)
    
    if s["1"].max() >= 0.8:
        s = s.sort_values(by = "1", ascending = False)
        context_sentences  = list(s[0].head(n_answers).values)
          
        for context in context_sentences:
            ans = get_answer(question, context)
            if len(ans) > 0:
                answers.append(ans)
    else:
        answers = get_wh_answer2(question, passage_embedding, passages, n_answers) 
    
    if len(answers) == 0:
        doc = nlp(context_sentences[0])
        context = question.lower().split()
        for ent in doc.ents:
            if ent.label != None and ent.text.lower() not in context:
                answers.append(ent.text)
        for ent in doc.ents:
            if ent.label == None:
                answers.append(ent.text)
            
    return answers[:n_answers]
 
def get_wh_answer2(question, passage_embedding, passages, n_answers = 1):
    q_emb  = bi_encoder.encode(question)
    context_sentences =  get_context(question, passage_embedding, passages, n_answers)
    pred_answers = []
    for context in context_sentences:
        ans = get_answer(question, context)
        if len(ans) > 0:
            pred_answers.append(ans)
    return pred_answers

#, contexts

#sentences = enumerate(sent_tokenize(txt))
#global passages 
#passages = [[str(index), sentence] for index, sentence in sentences]
#passage_embedding = bi_encoder.encode(passages)
#p = pd.DataFrame(passages)
#contexts = []
#q_emb  = bi_encoder.encode(question)
#p[2] = sim(passage_embedding, q_emb.reshape(1,-1))
#context_sentences = get_context(question, passage_embedding)
#q_emb  = bi_encoder.encode(question)
#p[2] = sim(passage_embedding, q_emb.reshape(1,-1))
#context_sentences = get_context(question, passage_embedding)

    #print(type(context_sentences[0]))
#cont = []
#ans = []
        #cont.append(context)

        #contexts.append(cont)
#answers.append(ans)
#print("Question: ", question, "\nContext: ",context[0].replace("\n",""),"\nAnswer: ", get_answer(question, context[0]))
#print()
        