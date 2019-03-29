import os
import pytest
import sys
import tempfile
from flask import url_for
sys.path.append('/home/ayushmaan/Desktop/WIT_Complete')
from app import app

@pytest.fixture
def client():
    db_fd, app.config['DATABASE'] = tempfile.mkstemp()
    app.config['TESTING'] = True
    client = app.test_client()

    yield client

    os.close(db_fd)
    os.unlink(app.config['DATABASE'])


def login(client,username,password):
    return client.post('/login',data=dict(
        username=username,
        password=password
    ), follow_redirects=True)

def test_empty(client):
    """Start with a blank database."""

    rv = client.get('/')
    assert b'html' in rv.data

def test_login_default(client):
    rv = login(client, 'admin', 'admin')
    assert str(rv.status) == "200 OK"

def test_login_empty_password(client):
    rv = login(client,'admin',' ')
    assert str(rv.status) == "200 OK"
    assert b'Log in' in rv.data
    assert b'interactive' not in rv.data

def test_login_empty_username(client):
    rv = login(client,' ','admin')
    assert str(rv.status) == "200 OK"
    assert b'Log in' in rv.data
    assert b'interactive' not in rv.data

def test_login_empty_both(client):
    rv = login(client,' ',' ')
    assert str(rv.status) == "200 OK"
    assert b'Log in' in rv.data
    assert b'interactive' not in rv.data
