#!/usr/bin/env python
import argparse
import sys
from qg.whqg.whqg_main import *


def get_wh_qsts(ARTICLE_FILE, NQUESTIONS):
    questions = get_wh_questions(ARTICLE_FILE, NQUESTIONS)
    return questions

if __name__ == "__main__":
    ARTICLE_FILE = sys.argv[1]
    NQUESTIONS = int(sys.argv[2])
    questions = get_wh_qsts(ARTICLE_FILE, NQUESTIONS)
    for question in questions:
        print(question)