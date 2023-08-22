from flask import Flask,request,render_template
import pickle

app=Flask(__name__)


@app.route('/')
def hello():
    return(render_template('index.html'))


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    position=data[0]
    player=data[1]
    if(position=='FWD'):
        df=pickle.load(open('FWD.pkl','rb'))
    elif(position=='MID'):
        df=pickle.load(open('MID.pkl','rb'))
    elif(position=='DEF'):
        df=pickle.load(open('DEF.pkl','rb'))
    else:
        df=pickle.load(open('GK.pkl','rb'))
    response=df[df['Name']==player]['Predicted_points']
    if(response.empty==False):
        response=round(float(response),2)
        return render_template('index.html',pred='Predicted points for '+player+' : {}'.format(response))
    else:
        return render_template('index.html',pred='Error :Player doesnt exist / Wont play next GW')




if __name__ == '__main__':
    app.run(debug=True)
 