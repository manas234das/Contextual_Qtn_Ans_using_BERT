from flask_wtf import FlaskForm
from wtforms import TextAreaField, StringField, SubmitField
from wtforms.validators import DataRequired

class Open_textArea(FlaskForm):

    # Getting the paragraph input
    paragraph = TextAreaField('Paragraph', validators=[DataRequired()])

    # Getting the question input
    question = StringField('Question', validators=[DataRequired()])

    submit = SubmitField('Generate answer')

