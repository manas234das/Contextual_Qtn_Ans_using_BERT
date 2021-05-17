import hashlib
from random import randint
def create_text_splits(text):
    return [text]

def create_single_example(paragraph,question):
    """Generates a topic question set as per squad format for question answering by Bert
    
    Arguments:
        paragraph {string} -- context paragraph to answer question from
        question {string} -- question asked
    
    Returns:
        [dict] -- a single question answer pair as per squad format
    """
    fake_single_answer = {"answer_start":0,
                     "text":"0"}
    fake_answer = [fake_single_answer,fake_single_answer,fake_single_answer]
    
    single_example = {}
    single_example['context'] = paragraph
    id = str(hashlib.md5((paragraph+question).encode()).hexdigest())+str(randint(0, 999999999))
    
    qas_dict = {}
    qas_dict['answers'] = fake_answer
    qas_dict['question'] = question
    qas_dict['id'] = id
    
    
    single_example['qas'] = [qas_dict]
    return single_example


def create_all_examples_for_one_result(id,text,question):
    list_of_paragraphs = create_text_splits(text)
    examples_list = []
    for para in list_of_paragraphs:
        examples_list.append(create_single_example(para,question))
        
    full_dict = {}
    full_dict['title'] = id
    full_dict['paragraphs'] = examples_list
    
    return full_dict


def create_all_examples_for_all_results(text_list,question,path_to_save):
    """For a single question, generates all the question answer pair from all paragraphs
    
    Arguments:
        text_list {list} -- List of paragraphs
        question {string} -- Question string
        path_to_save {[type]} -- Path to save generated json file to.
    """
    example_list = []
    for text in text_list:
        example_list.append(create_all_examples_for_one_result((hashlib.md5(text.encode()).hexdigest())+str(randint(0, 999999999)),
                                                               text,question))
    final = {}
    final["data"] = example_list
    import json
    with open(path_to_save+"//"+'dev-v1.1.json', 'w') as outfile:
        json.dump(final, outfile)




