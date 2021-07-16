import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import random
    import spacy
    import dill as pickle
    import pandas as pd
    import nltk
    from nltk import sent_tokenize,word_tokenize
    import numpy as np
    from qg.spacy_clausie import claucy #import claucy
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelWithLMHead
    import random

def paraphrase(s):
    text =  "paraphrase: " + s + " </s>"

    encoding = tokenizer_p.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outputs = model_p.generate(
        input_ids = input_ids, attention_mask = attention_masks,
        max_length = 256,
        do_sample = True,
        top_k = 200,
        top_p = 0.95,
        early_stopping = True,
        num_return_sequences = 1
    )
    
    lines = []
    for output in outputs:
        line = tokenizer_p.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        lines.append(line)
        
    line = random.choice(lines)
    #print(line)
    return line

try:
    nlp = spacy.load("en_core_web_lg")
except:
    try:
         nlp = spacy.load("en_core_web_md")
    except:
        nlp =  spacy.load("en_core_web_sm")
        
tokenizer_t = AutoTokenizer.from_pretrained('./qg/finetuned_qg_model/')
model_t = AutoModelWithLMHead.from_pretrained('./qg/finetuned_qg_model/')

def extract_ner_hf(txt):
    result = hf(txt)
    ents = {}
    for ent in result:
        ents[ent['word']] = ent['entity_group']
    return ents

def get_sentences(txt):
    sentences = sent_tokenize(txt)
    return sentences

def get_ents(txt):
    ents1 = {}
    doc = get_doc(txt)
    sentences = get_sentences(txt)
    
    for ent in doc.ents:
        ents1[ent.text] = ent.label_
    
    ents2 = {}
     
    ents3 = {}
    
    return sentences, ents1, ents2, ents3
          
    
def generate_questions_ent(txt):
    sentences, entities1, entities2, entities3 = get_ents(txt)
    entities = list(set(list(entities1.keys()) + list(entities2.keys()) + list(entities3.keys())))
    questions = []
    for ent in entities:
        if entities1.get(ent) in ["PERSON", "NORP"] or entities2.get(ent) in ["PERSON", "NORP"] or entities3.get(ent) == "PER":
            if nltk.pos_tag([ent])[0][1] in ["NNS", "NNPS"]:
                questions.append("Who are {}?".format(ent))
            else:
                questions.append("Who is {}?".format(ent))
        elif entities1.get(ent) in ["GPE", "LOC", "FAC"] or entities2.get(ent) in ["GPE", "LOC", "FAC"] or  entities3.get(ent) == "LOC":
                questions.append("Where is {}?".format(ent))
        elif entities1.get(ent) in ["DATE", "TIME"] or entities2.get(ent) in ["DATE", "TIME"]:
            questions.append("What happened in {}?".format(ent))
        elif entities1.get(ent) == "EVENT" or entities2.get(ent) == "EVENT":
            questions.append("What happened at {}?".format(ent))
            questions.append("When did {} happen?".format(ent))
        elif entities1.get(ent) == "ORDINAL" or entities2.get(ent) == "ORDINAL":
             questions.append("What happened for the {} time?".format(ent))
        elif entities1.get(ent) in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
             questions.append("What is {}?".format(ent))                  
    return questions, sentences, entities1, entities2, entities3

def return_tag(sentence):
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return " ".join([tag[1] for tag in tagged])

def get_doc(txt):
    doc = nlp(txt)
    return doc

def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer_t([input_text], return_tensors='pt')

  output = model_t.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer_t.decode(output[0])

def generate_questions_transformer(sentences, n):
    qs = []
    random.seed(123)
    sentences = random.sample(sentences, n)
    for i in range(len(sentences)):
        doc = get_doc(sentences[i])
        for ent in doc.ents:
                  if ent.label_ in ['PERSON', 'DATE', 'GPE', 'LOC', 'ORG', 'EVENT']:
                      context = sentences[i]
                      #print(context)
                      qs.append([get_question(ent, context), ent])
    r_qsts = set()
    for qq in qs:
        xx = qq[0]
        r_qsts.add(xx[xx.find(":")+2:-4])
    r_qsts = list(r_qsts)
    return r_qsts

def string(l):
    ad = ""
    for i in l.adverbials:
        ad = ad + str(i) + " "
    a = str(l.type) + "~" +str(l.subject) + "~" + str(l.verb) + "~" + str(l.indirect_object)\
    + "~" + str(l.direct_object) \
    + "~" + str(l.complement) + "~" + ad
    return a

def get_triples(sentences):
    breakdown = []
    nlp3 = spacy.load('en_core_web_lg')
    claucy.add_to_pipe(nlp3)
    for sentence in sentences:
        #length = len(sentence.split())
        #sentence = summarizer(sentence, max_length = int(0.8 * length), min_length = int(0.5 * length), do_sample=False)[0]['summary_text']
        doc = nlp3(sentence) 
        breakdown = breakdown + doc._.clauses
    return breakdown

def return_questions_ie(sentences, ents, ents2, ents3):
    breakdown = get_triples(sentences)
    ie_data = pd.DataFrame(breakdown)[0].apply(string).str.split("~", expand = True).rename(columns={0:'type',
                                                                                          1:'subject',
                                                                                          2:'verb',
                                                                                          3:'indirect_obj',
                                                                                          4:'direct_obj',
                                                                                          5:'complement',
                                                                                          6:'adverbials'})
    ie_data = ie_data.replace("None", "")

    que = []
    for row in ie_data.iterrows():
        if "NN" not in return_tag(row[1][1]) or row[1][2] == "":
            continue
        if ents.get(row[1][1]) in ["NORP","PERSON"] or ents2.get(row[1][1]) in ["NORP","PERSON"] or ents3.get(row[1][1]) == "PER":
            tag = "Who "
        elif ents.get(row[1][1]) in ["DATE", "TIME"] or ents2.get(row[1][1]) in ["DATE", "TIME"]:
            tag = "When "  
        elif ents.get(row[1][1]) in ["GPE", "LOC", "FAC"] or ents2.get(row[1][1]) in ["GPE", "LOC", "FAC"]:
            tag = "Where "
        else:
            tag = "What "
            
        end = row[1][4]
        if end == "":
            end = row[1][5]
        if end == "":
            end = row[1][6]
        if end == "":
            continue
            
        q = tag + row[1][2] + " " + row[1][3] + " " + end + " ?"
        keys = set(ents.keys())
        #_, e, _, _ = get_ents(q)
        for word in word_tokenize(q):
            if word in keys:
                #q = paraphrase(q)
                que.append(q)
                break
    return que
