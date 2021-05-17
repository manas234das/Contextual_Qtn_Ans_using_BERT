"""
Note:
1. Make sure that you download the 'en_core_web_sm' and all the required libraries.
   There are other versions available as well. If you want you can use the other versions as well.
2. There is a symbol to text conversion method. Its by default commented out. 
   If you want you can simply uncomment the section.
"""

from nltk.tokenize import sent_tokenize, word_tokenize
import inflect
import spacy
import nltk
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_short
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import remove_stopwords
# Loading the spacy module
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def replace_numbers(text):
    """
    Replace all interger occurrences in list of tokenized words with textual representation
    Args : { text: raw sentence }
    returns : { list of words with numbers converted to english meaning }
    """
    words = word_tokenize(text)
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

# def symbol_conversion(text):
#     """
#     Replace all symbols occurrences in list of tokenized words with textual representation
#     Args : { text: raw sentence }
#     returns : { list of symbols with numbers converted to english meaning }
#     """
#     symbols_dict = {'equal':'=', 'of x':'(x)','greater than or equal to':'≥',
#                     'less than or equal to':'≤','less than':'<','greater than':'>',
#                     'Fig':'figure','plus':'+','minus':'-','multiply':'*','multiply':'×','divide':'/'}
#     words = word_tokenize(text)
#     for i, word in enumerate(words):
#         for k,v in symbols_dict.items():
#             if word == v:
#                 words[i] = k
#             else:
#                 pass
    
#     return words
        
def data_preprocessing(para):
    """
    This function takes in paragraph and returns a list pre-processed sentences

    Args : { para: raw paragraph }
    returns : { list of individual sentences in the paragraph }
    """
    # Splitting the paragraph into sentences
    sentences = sent_tokenize(para)
    processed_sentences = []
    for sent in sentences:
        # lowercase
        temp_text = sent.lower()
        
        # Converting sybols
        # temp_text = " ".join(symbol_conversion(sent))
        
        # Removing the non alphabetic symbols
        temp_text = strip_non_alphanum(sent)
        # Removing multiple white spaces
        temp_text = strip_multiple_whitespaces(temp_text)
        # Removing punctuations
        temp_text = strip_punctuation(temp_text)
        # Converting digits to alphabets
        temp_text = " ".join(replace_numbers(temp_text))
        
        # Remove stopword
        # temp_text = remove_stopwords(temp_text)
        
        # Remove short 1 letter values
        temp_text = strip_short(temp_text, minsize=2)
        
        # Lemmatization
        # doc = nlp(temp_text)
        # temp_text = " ".join([token.lemma_ for token in doc])
        
        if len(temp_text) > 1:
            processed_sentences.append(temp_text.lower())
    
    return processed_sentences



