import os
import pytest
import sys
import tempfile
from flask import url_for
sys.path.append('/home/ayushmaan/Desktop/WIT_Complete')
from app import app
from flask_testing import TestCase
import io
import flask
class TemplateTest(TestCase):

    def create_app(self):
        return app

    def test_wrong_file_format(self):
        with open('/home/ayushmaan/Desktop/heart.csv','rb') as csv_file:
            data = {}
            data['Labels'] = "age,sex,cp,trestbps,chol,fbs,thalach,exang,oldpeak,slope,ca,thal,target"
            data['Target'] = "samp;e"
            data['Zero_Value'] = "0"
            data['file'] = (csv_file,'heart.csv')
            rv = self.app.test_client().post('/uploadForm',
                                                data=data,
                                                content_type='multipart/form-data')

            assert b"Error" not in rv.data

        
