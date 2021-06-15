```python
%%time
import argparse
import sys
from qg.whqg.whqg_main import *
from qa.wh_qa.wh_qa import *

def read_file(file):
    """
    Read text from file
    :param file: input file path
    :return: text - text content in input file
    """
    with open(file, encoding = 'utf-8', mode = 'r') as f:
        text = f.readlines()
    return text

def get_wh_ans(ARTICLE_FILE, QUESTION_FILE):
    answers = []
    passage_embedding, passages, sentences = get_embs(ARTICLE_FILE)
    questions = read_file(QUESTION_FILE)
    
    for question in questions:
        ans = get_wh_answer(sentences, question, passage_embedding, passages, 1)
        answers.append(ans[0])
    return answers

def get_wh_qsts(ARTICLE_FILE, NQUESTIONS):
    questions = get_wh_questions(ARTICLE_FILE, NQUESTIONS)
    return questions

print("***************************************************************************")
questions = get_wh_questions("art.txt", 50)
for question in questions:
    print(question)
print("***************************************************************************")
answers = get_wh_ans("art.txt", "art_questions.txt")
for answer in answers:
    print(answer)
print("***************************************************************************")
```

    INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: ./qa/nq-distilbert-base-v1
    INFO:sentence_transformers.SentenceTransformer:Load SentenceTransformer from folder: ./qa/nq-distilbert-base-v1
    

    ***************************************************************************
    In what century did art remain a marker of wealth and status?
    In what year was the debate on the relationship between conceptual and perceptual encounters with art held?
    What been described  by philosopher Richard Wollheim as "one of the most elusive of the traditional problems of human culture"  ?
    In what decade did artists expand the technique of self-criticism?
    What is the name of the publisher of the University of Chicago Press?
    What is  the Silk Road, where Hellenistic, Iranian, Indian and Chinese influences could mix ?
    Who gives  no hint of the disapproval of Homer that he expresses in the Republic ?
    What suggests  that Homer's Iliad functioned in the ancient Greek world as the Bible does today in the modern Christian world: as divinely inspired literary art that can provide moral guidance, if only it can be properly interpreted ?
    What is the name of the artist who was featured in Artforum?
    What is the title of the book by Aut and Livingston?
    Until what century was art not differentiated from crafts?
    In what decade did artists expand self-criticism?
    What is the title of the book Elkins wrote?
    Who is the author of Essay VI?
    In what region were cylindrical seals widely used?
    What is the name of the citation?
    What is  the multicultural port metropolis of Trieste ?
    What believed  that imitation is natural to mankind and constitutes one of mankind's advantages over animals ?
    What opened  the first public museum of art in the world, the Kunstmuseum Basel ?
    What impelled  the aesthetic innovation which germinated in the mid-1960s and was reaped throughout the 1970s ?
    What saw  an equivalent influence of other cultures into Western art ?
    What depicts  a female nude, hooded detainee strapped to a chair, her legs open to reveal her sexual organs, surrounded by two tormentors dressed in everyday clothing ?
    What published  a classic and controversial New Critical essay entitled "Intentional Fallacy|The Intentional Fallacy", in which they argued strongly against the relevance of an Authorial intentionality|author's intention, or "intended meaning" in the analysis of a literary work ?
    What is another name for objectivity?
    Who wrote The Invention of Art: A Cultural History?
    Where saw  the flourishing of many art forms: jade carving, bronzework, pottery (including the stunning terracotta army of Emperor QinGombrich, pp ?
    What is the philosophy of Wollheim?
    What became  a major industry from the Renaissance ?
    Who was the aesthetic theorist who championed what he saw as the naturalism of J.M.W. Turner?
    When did the US begin to see the end of the financial crisis?
    When was the Observer's "In bed with Tracey, Sarah... and Ron" published?
    Who wrote the book "Philosophy for Architecture"?
    What is  a photograph of a crucifix, sacred to the Christian religion and representing Jesus Christ|Christ's sacrifice and final suffering, submerged in a glass of the artist's own urine ?
    When did the West have a huge impact on Eastern art?
    What era had a huge impact on Eastern art?
    Where is the article "In bed with Tracey, Sarah... and Ron"?
    When was Martin Heidegger's "The Origin of the Work of Art" published?
    What can be understood neither as art or craft?
    What include  an idea of imaginative or technical skill stemming from Agency (philosophy)|human agency and creation ?
    What website published the article Asger Jorn?
    What necessitates  a re-evaluation of aesthetic theory in art history today and a reconsideration of the limits of human creativity ?
    What explained  an additional connection between the destruction of cultural property and the cause of flight during a mission in Lebanon in April 2019 ?
    Where was Martin Heidegger's book published?
    Where was Introduction to Structuralism published?
    When was Introduction to Structuralism published?
    Along with Hamann, what other philosopher was responsible for the development of the philosophy of language?
    What persisted  Nevertheless in small Byzantine works  ?
    When was Asger Jorn at Artforum?
    What advanced  the Idealism|idealist view that art expresses emotions, and that the work of art therefore essentially exists in the mind of the creator ?
    Who wrote Introduction to Structuralism?
    ***************************************************************************
    


    Batches:   0%|          | 0/11 [00:00<?, ?it/s]


    21st
    art, n. 1
    formalism
    2000s
    The University of Chicago Press Books
    the silk road, where hellenistic, iranian, indian and chinese influences could mix? [SEP] an example of this is the silk road
    socrates
    divinely inspired literary art that can provide moral guidance, if only it can be properly interpreted
    zeitgeist
    gaut and livingston, p. 6
    17th
    2000s
    napoleon
    w. k. wimsatt | william k. wimsatt and monroe beardsley
    ancient near east
    zeitgeist
    trieste
    aristotle
    1661
    a work of art
    global interaction
    interrogation iii
    intentional fallacy | the intentional fallacy ", in which they argued strongly against the relevance of an authorial intentionality | author's intention, or " intended meaning " in the analysis of a literary work? [SEP] in 1946, w. k. wimsatt | william k. wimsatt and monroe beardsley
    philosophy
    the invention of art : a cultural history? [SEP] the invention of art : a cultural history.
    where
    aesthetic quality is an absolute value independent of any human view
    Italy
    john ruskin
    there is evidence that there may be an element of truth
    20 april 2003
    branco mitrovic
    christ
    19th and 20th centuries
    post - modernism
    the observer
    when was martin heidegger's " the origin of the work of art " published? [SEP] in the origin of the work of art, martin heidegger, a german philosopher and a seminal thinker, describes the essence of art in terms of the concepts of being and truth.
    techne
    general descriptions
    asger jorn
    artificial intelligence
    karl von habsburg
    harper perennial
    university of michigan
    1970
    kant
    a classical realist tradition
    1 september 2001
    Benedetto Croce
    michael lane
    ***************************************************************************
    Wall time: 2min 20s
   
