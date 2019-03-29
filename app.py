from flask import Flask
from flask import render_template,flash,redirect,request,url_for
from forms import LoginForm, DataForm,TestForm,ExplainForm
from config import Config
import subprocess
import sys
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
from upload import upload_for_tensorboard,run_model,run_protobuf_model
from explanations import run_explanations

UPLOAD_FOLDER = '/home/datasets'
MODEL_FOLDER = '/home/ayushmaan/trained_model'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
@app.route('/index')
def index():
    command = "docker kill $(docker ps -q)"
    os.system(command)
    user = {'username':'user'}
    return render_template('index.html',user=user)


@app.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        #flash('usernameis:{},remember it:{}'.format(
            #form.username.data,form.remember_me.data))
        #v = sys.version
        #flash(v)
        return redirect('/index')
    return render_template('login.html',title='Log in',form=form)

@app.route('/test',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #temp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
    return render_template('test.html')

@app.route('/uploadForm',methods=['GET','POST'])
def uploadFile():
    form = TestForm()
    if request.method == 'POST':
        command = "docker kill $(docker ps -q)"
        os.system(command)
        labels = form.Labels.data
        labels = labels.replace(' ','')
        labels = labels.split(',')
        target = form.Target.data
        zero_value = form.Zero_Value.data
        #if str.isdecimal(zero_value):
        #    zero_value = int(zero_value)

        print("Type of zero value is ",type(zero_value))
        print(labels,target)
        filename = secure_filename(form.file.data.filename)
        form.file.data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #uploaded_file(filename,labels,target,zero_value)
        model_path = MODEL_FOLDER

        what_if_path = run_model(os.path.join(app.config['UPLOAD_FOLDER'],filename),model_path,labels,target,zero_value)
        if what_if_path == "target column error":
            flash("Target Column Error")
            return redirect(request.url)
            #return render_template('upload.html',title='Form Uploader',form=form)
        elif what_if_path == "zero value error":
            flash("Zero Value Error")
            return redirect(request.url)
            #return render_template('upload.html',title='Form Uploader',form=form)
        else:
            #model_path = MODEL_FOLDER
            #what_if_path = run_model(os.path.join(app.config['UPLOAD_FOLDER'],filename),model_path,labels,target,zero_value)
            return redirect(what_if_path)
    return render_template('upload.html',title='Form Uploader',form=form)

@app.route('/uploads/<filename>')
def uploaded_file(filename,labels,target,zero_value):
    return redirect('/')

@app.route('/auditor',methods=['GET','POST'])
def audit_model():
    form = TestForm()
    if request.method == 'POST':

        labels = form.Labels.data
        labels = labels.replace(' ','')
        labels = labels.split(',')
        target = form.Target.data
        zero_value = form.Zero_Value.data

        print("Type of zero value is ",type(zero_value))
        print(labels,target)
        filename = secure_filename(form.file.data.filename)
        form.file.data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        explanation_return = run_explanations(os.path.join(app.config['UPLOAD_FOLDER'],filename),labels,target,zero_value)

        if explanation_return == "target column error":
            flash("Target Column Error")
            return redirect(request.url)
            #return render_template('upload.html',title='Form Uploader',form=form)
        elif explanation_return == "zero value error":
            flash("Zero Value Error")
            return redirect(request.url)

        else:
            return render_template('explanation.html')

    return render_template('upload2.html',title='Form Uploader',form=form)

@app.route('/explain',methods=['GET','POST'])
def explainFile():
    form = ExplainForm()
    if request.method == 'POST':
        command = "docker kill $(docker ps -q)"
        os.system(command)
        labels = form.Labels.data
        labels = labels.replace(' ','')
        labels = labels.split(',')
        target = form.Target.data
        zero_value = form.Zero_Value.data
        model_path = form.Model_Folder.data
        #if str.isdecimal(zero_value):
        #    zero_value = int(zero_value)

        print("Type of zero value is ",type(zero_value))
        print(labels,target)
        filename = secure_filename(form.file.data.filename)
        form.file.data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #uploaded_file(filename,labels,target,zero_value)
        #model_path = MODEL_FOLDER

        what_if_path = run_protobuf_model(os.path.join(app.config['UPLOAD_FOLDER'],filename),model_path,labels,target,zero_value)
        if what_if_path == "target column error":
            flash("Target Column Error")
            return redirect(request.url)
            #return render_template('upload.html',title='Form Uploader',form=form)
        elif what_if_path == "zero value error":
            flash("Zero Value Error")
            return redirect(request.url)
            #return render_template('upload.html',title='Form Uploader',form=form)
        else:
            #model_path = MODEL_FOLDER
            #what_if_path = run_model(os.path.join(app.config['UPLOAD_FOLDER'],filename),model_path,labels,target,zero_value)
            return redirect(what_if_path)
    return render_template('explain.html',title='Form Uploader',form=form)



if __name__ == '__main__':
    app.run()
