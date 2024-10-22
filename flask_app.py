from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

app = Flask(__name__)

with open(r"toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

with open(r"toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/classify", methods=['POST'])
def predict():
    
    user_input = request.form['text']
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1]

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1]

    output_tox = round(pred_tox[0], 2)
    output_sev = round(pred_sev[0], 2)
    output_obs = round(pred_obs[0], 2)
    output_ins = round(pred_ins[0], 2)
    output_thr = round(pred_thr[0], 2)
    output_ide = round(pred_ide[0], 2)

    print(out_tox)

    return render_template('index_toxic.html', 
                            pred_tox = 'Toxic : {}'.format(output_tox),
                            pred_sev = 'Severely Toxic : {}'.format(output_sev), 
                            pred_obs = 'Obscene: {}'.format(output_obs),
                            pred_ins = 'Insulting : {}'.format(output_ins),
                            pred_thr = 'Threatening : {}'.format(output_thr),
                            pred_ide = 'Identity Hate Related: {}'.format(output_ide)                        
                            )
     

app.run(debug=True)

