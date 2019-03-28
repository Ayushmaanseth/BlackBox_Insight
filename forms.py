from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed, FileRequired
from wtforms import StringField,PasswordField,BooleanField,SubmitField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField('username',validators=[DataRequired(message='Input username...')])
    password = PasswordField('password',validators=[DataRequired(message='Input password...')])
    remember_me = BooleanField('Remember me')
    submit = SubmitField('Log in')


class DataForm(FlaskForm):
    data1 = StringField('data1',validators=[DataRequired(message='Input data1...')])
    data2 = StringField('data2',validators=[DataRequired(message='Input data2...')])
    submit = SubmitField('Evaluate')

class TestForm(FlaskForm):
    Labels = StringField('Labels',validators=[DataRequired(message="Input labels separated by comma")])
    Target = StringField('Target',validators=[DataRequired(message="Enter the target column")])
    Zero_Value = StringField('Zero Value',validators=[DataRequired(message="Enter the target value corresponding to prediction = 0")])
    file = FileField('CSV File',validators=[FileRequired(),FileAllowed(['csv'])])
    submit = SubmitField('Submit Form')


class ExplainForm(FlaskForm):
    Labels = StringField('Labels',validators=[DataRequired(message="Input labels separated by comma")])
    Target = StringField('Target',validators=[DataRequired(message="Enter the target column")])
    Zero_Value = StringField('Zero Value',validators=[DataRequired(message="Enter the target value corresponding to prediction = 0")])
    Model_Folder = StringField('Model Folder',validators=[DataRequired(message="Enter the model folder path")])
    file = FileField('CSV File',validators=[FileRequired(),FileAllowed(['csv'])])
    submit = SubmitField('Submit Form')
