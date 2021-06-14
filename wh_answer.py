#!/usr/bin/env python3
import argparse
import sys
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


# Main function doesn't need to be modified
#if __name__ == "__main__":
#    ARTICLE_FILE = sys.argv[1]
#    QUESTION_FILE = sys.argv[2]
#    
#    answers = get_wh_ans(ARTICLE_FILE, QUESTION_FILE)
#    for answer in answers:
#        print(answer)