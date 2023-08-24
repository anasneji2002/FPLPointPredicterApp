import os
import subprocess
from flask import Flask,request,render_template
import pickle
import pandas as pd

app=Flask(__name__)

def delete_pickle_file(position):
    filename = position + '.pkl'
    if os.path.exists(filename):
        os.remove(filename)

@app.route('/')
def hello():
    return(render_template('index.html'))


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    position=data[0]
    player=data[1]
    delete_pickle_file(position)
    if(position=='FWD'):
        subprocess.run(['python', 'FWD.py'], shell=True)
        df = pd.read_pickle('FWD.pkl')
    elif(position=='MID'):
        subprocess.run(['python', 'MID.py'], shell=True)
        df = pd.read_pickle('MID.pkl')
    elif(position=='DEF'):
        subprocess.run(['python', 'DEF.py'], shell=True)
        df = pd.read_pickle('DEF.pkl')
    else:
        subprocess.run(['python', 'GK.py'], shell=True)
        df = pd.read_pickle('GK.pkl')
    response=df[df['Name']==player]['Predicted_points']
    if(response.empty==False):
        response=round(float(response),2)
        return render_template('index.html',pred='Predicted points for '+player+' : {}'.format(response))
    else:
        return render_template('index.html',pred='Error :Player doesnt exist / Wont play next GW')




if __name__ == '__main__':
    app.run(debug=True)
 