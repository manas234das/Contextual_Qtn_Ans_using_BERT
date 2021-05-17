"""
This is the main file which loads the pre-trained models and returns the answers.
Note:
    1.  This module uses deeppavlov's pre-trained model. It's because it gave vetter results than the 
        BERT base model. For more details check out ( http://docs.deeppavlov.ai/en/master/ )
    2.  Make sure that deeppavlov is installed beforing running the scripts.
        It will download 2.5GB of additional files. 
    3.  The data used in this module is a excell file containing 4 columns.
        CHAPTER_NUMBER, CHAPTER NAME, HEADING, SUB-HEADING, CONTENT
        - To use this module with some other PDFs, extract the contents of the PDFs using 
          Allen AI Science-Parser ( https://github.com/allenai/science-parse ) 
          which creates sections just like this.
"""

import pandas as pd
from subParas import samplePara
from generate_dev import create_all_examples_for_all_results
from flask import Flask, render_template, url_for, flash, redirect, request
from openQA import Open_textArea
import subprocess
import json
import torch
from flask import Markup
from deeppavlov import build_model, configs
from nltk.tokenize import sent_tokenize
import re
import os
import math
from BERT_PYTORCH_MODIFIED.Workers.Classifier import train
from BERT_PYTORCH_MODIFIED.pytorch_pretrained_bert.modeling import BertModel,BertForSequenceClassification

# If KMP_AFFINITY is activated the processing will be slower
os.environ["KMP_AFFINITY"] = "none"

# Creating the app
app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

# Load the model (deeppavlov BERT)
model_bert = build_model(configs.squad.squad, download=True)

# Loading the sentence classification model
model = BertForSequenceClassification.from_pretrained("bert-base-cased")
model.load_state_dict(torch.load("model.ckpt",map_location='cpu'))

# Importing thed data
project_data = pd.read_excel("data/Engineering_Optimization.xlsx")
temp_df = project_data.drop_duplicates(['chapter_name', 'chapter_number'])

# Creating the Base API. This will be the 1st API to be called when the app runs.
@app.route("/")
@app.route("/home")
def home():
    chapters_nn = []
    for i,k in enumerate(temp_df['chapter_name'].values):
        chapters_nn.append({f"chapter_number": f"{int(i+1)}", f"chapter_name": f"{k}"})

    return render_template('home.html', chapters_nn = chapters_nn)

# This is the next API where the chapter's questions along with the content is being displayed.
@app.route("/chapter/<chapter_num>")
def chapter(chapter_num):
    
    # Getting the questions and paragraphs of all the chapters
    all_review_questions_df = project_data[project_data["heading"] == "REVIEW QUESTIONS"]

    # Get chapter input
    chapter = int(chapter_num)

    # Specific chapter's questions
    questions_list = all_review_questions_df[all_review_questions_df["chapter_number"] == chapter]["content"].values.tolist()[0].split("\n")
    paragraph_list = project_data[project_data["chapter_number"] == chapter]["content"].values.tolist()
    paragraph_list = paragraph_list[0:len(paragraph_list)-1]
    heading_list = project_data[project_data["chapter_number"] == chapter]["heading"].values.tolist()
    heading_list = heading_list[0:len(heading_list)-1]
    sub_heading_list = project_data[project_data["chapter_number"] == chapter]["sub_heading"].values.tolist()
    sub_heading_list = sub_heading_list[0:len(sub_heading_list)-1]

    for j, y in enumerate(sub_heading_list):
        if y == 'NONE':
            sub_heading_list[j] = heading_list[j]

    para_head = {}
    for i,k in enumerate(paragraph_list):
        para_head[sub_heading_list[i]] = k
        
    para_head_list = []
    for i,k in enumerate(paragraph_list):
        para_head_list.append({f"heading": f"{list(para_head.keys())[i]}", f"paragraph" : f"{list(para_head.values())[i]}"})

    return render_template("chapter.html", chapter_num = chapter_num, 
                            questions_list = questions_list, para_head_list = para_head_list)

# This API is used to get the questions from the click or from the text box.
# It runs BERT on that questions and generates the answers.
@app.route("/question/<qtn><chapter_num>", methods=['GET','POST'])
def question(qtn, chapter_num):

    # Get chapter and question input
    chapter = int(chapter_num)
    question = str(qtn)
    print(question)
    if qtn == 'Extra_qtn':
        question = request.form['ext_qtn']
        print(question)

    # Getting the paragraphs
    paragraph_list = project_data[project_data["chapter_number"] == chapter]["content"].values.tolist()
    paragraph_list = paragraph_list[0:len(paragraph_list)-1]

    predictions = {}

    # Using the BERT model to get the answers
    for i, para in enumerate(paragraph_list):
        pred = model_bert([para],[question])
        predictions[i] = pred[0][0]
    
    # Writing into the file
    with open('./results/predictions.json', 'w') as fp:
        json.dump(predictions, fp)

    # Opening the predictions
    with open('./results/predictions.json') as f:
        data = json.load(f)
        
    snippet_list = []
    for value in data.values():
        snippet_list.append([question, value])

    # Formatting results to show the highlight part of the answer
    for i, para in enumerate(paragraph_list):
        # Tokenizing the sentences
        sentences = sent_tokenize(para)
        for j, sent in enumerate(sentences):
            snip = str(snippet_list[i][1])
            query = r"\b" + re.escape(snip) + r"\b"
            # Searching for the snippet
            if re.search(query, sent, re.IGNORECASE):
                ans_sent = sent
                snippet = Markup("<mark class='highlight'>{}</mark>".format(ans_sent))
                paragraph_list[i] = paragraph_list[i].replace(ans_sent, snippet)

    # Query relevance part, finding the order of relevance of the answers
    pred_list = []
    for result in paragraph_list:
        pred_list.append([str(0),str(0),str(0), question, result])        
    
    predicts = train(model,'bert-base-cased', [], False,False,True,[],pred_list)
    temp = []
    for item in predicts:
        temp.append(item.tolist())
    
    prob_result_list = []
    for item, result in zip(temp, paragraph_list):
        prob = math.exp(item[1])/(math.exp(item[1])+math.exp(item[0]))
        prob_result_list.append([prob, result])

    #Step 7: Reverse sorting the results
    prob_result_list.sort(key=lambda x: x[0], reverse=True)

    result_list = [x for x in prob_result_list]

    return render_template('results.html', question=question, result_list=result_list)


@app.route("/openQA", methods=['GET', 'POST'])
def openQA():

    form = Open_textArea()
    if form.validate_on_submit():

        # question
        open_question = form.question.data
        # para
        open_para = form.paragraph.data
        # Using bert to answer the questions
        answer = model_bert([open_para],[open_question])[0][0]
        # Flashing the answer
        flash("Answer : {}".format(str(answer)), 'success')
        
        # return redirect(url_for('openQA'))

    return render_template("openQA.html", form=form)


if __name__ == '__main__':
    app.run(host='localhost', port=7850, debug=False)
