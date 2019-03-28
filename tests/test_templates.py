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

    def test_default_rendering(self):
        self.app.test_client().get('/')
        self.assert_template_used('index.html')

    def test_login_rendering(self):
        rv = self.app.test_client().post('/login',data=dict(
        username='admin',
        password='admin'
        ),follow_redirects=True)

        self.assert_template_used('login.html')
        self.assertEqual(rv.status,'200 OK')


    def test_counterfactual_rendering(self):
        self.app.test_client().get('/uploadForm')
        self.assert_template_used('upload.html')


    def test_audit_rendering(self):
        self.app.test_client().get('/auditor')
        self.assert_template_used('upload2.html')

    def test_tensorboard_rendering(self):
        with open('/home/ayushmaan/Desktop/adult.csv','rb') as csv_file:
            rv = self.app.test_client().post('/uploadForm',data=dict(
                Labels="Age,Workclass,fnlwgt,Education,Education-Num,Marital-Status,Occupation,Relationship,Race,Sex,Capital-Gain,Capital-Loss,Hours-per-week,Country,Target",
                Target="Target",
                Zero_Value="<=50K",
                file=(csv_file,csv_file.name)
            ),follow_redirects=True,content_type='multipart/form-data')

        self.assertEqual(rv.status,"200 OK")

    def test_tensorboard_rendering_2(self):
        with self.app.test_request_context('/uploadForm'):
            assert flask.request.path == '/uploadForm'
