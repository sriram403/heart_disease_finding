from flask import Flask,request,render_template,url_for
import pickle
import pandas as  pd

app = Flask(__name__)

with open("heart_model.pkl","rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict",methods=["Post"])
def predict():
    data = [float(x) for x in request.form.values()]
    print(data)
    yhat = model.predict([data])
    # yhat = 1
    classes = "Heart Disease" if yhat == 1 else "No heart disease"
    return render_template("index.html",prediction_text=f"You have a {classes}")

if __name__ == "__main__":
    app.run(debug=True)