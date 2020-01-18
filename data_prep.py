import json
import itertools
import re
from utils import get_path
from collections import Counter
from config import parameters, paths

vocabulary_dir = paths["vocabulary_dir"]
max_answers = parameters['max_answers']


def create_vocab(iterable, top_k=None, start=0):
    """
        Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """

    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        # Return the top k tokens as a vocabulary
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # Descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)

    # Create an enumerated vocabulary of the tokens
    vocabulary = {t: i for i, t in enumerate(tokens, start=start)}
    return vocabulary


# Some definitions for normalizing the questions and answers:
# This is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# These try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def normalize_questions(questions_json):
    """
        Tokenize and normalize questions from a given question json in the usual VQA format.
    """
    # The questions normalization is based on removing the special characters.
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        question = question.lower()[:-1]
        question = _special_chars.sub('', question)
        yield question.split(' ')


def normalize_answers(answers_json):
    """
        Normalize answers from a given answer json in the usual VQA format.
    """
    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    def process_punctuation(s):
        # Searches for punctuation characters and replaces them with empty chars or spaces
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()

    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


def prep_qa():
    """
        Loads the questions and answers from the jsons, normalizes and creates vocabularies
    """
    questions = get_path(train=True, question=True)
    answers = get_path(train=True, answer=True)

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    # Normalize the Q&A and create vocabularies
    questions = list(normalize_questions(questions))
    answers = list(normalize_answers(answers))
    question_vocab = create_vocab(questions, start=1)
    answer_vocab = create_vocab(answers, top_k=max_answers)

    vocabularies = {
        'question': question_vocab,
        'answer': answer_vocab,
    }

    # Saves the vocabulary as json in the defined path
    with open(vocabulary_dir, 'w') as fd:
        json.dump(vocabularies, fd)
