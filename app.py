from flask import Flask, render_template, request
import joblib
import os

from groq import Groq
# get key from .env file
# from dotenv import load_dotenv
# load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

client = Groq()

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

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    return render_template("chatbot.html")

@app.route("/llama", methods=["GET", "POST"])
def llama():
    return render_template("llama.html")

@app.route("/llama_result", methods=["GET", "POST"])
def llama_result():
    q = request.form.get("q")
    r = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q}
    ])
    r = r.choices[0].message.content
    return render_template("llama_result.html", r=r)

if __name__ == "__main__":
    app.run(debug=True)
