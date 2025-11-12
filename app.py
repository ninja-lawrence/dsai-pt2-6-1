from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/main", methods=["GET", "POST"])
def main():
    q = request.form.get("q")
    return render_template("main.html", q=q)

@app.route("/dbs", methods=["GET", "POST"])
def dbs():
    return render_template("dbs.html")

@app.route("/dbs_prediction", methods=["POST"])
def dbs_prediction():
    q = float(request.form.get("q"))
    model = joblib.load("DBS_SGD_model.pkl")
    r = model.predict([[q]])
    return render_template("dbs_prediction.html", r=r)

if __name__ == "__main__":
    app.run(debug=True)
