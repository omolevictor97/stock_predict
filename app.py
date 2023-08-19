from flask import Flask, render_template, request, url_for
import numpy as np
import pickle


model = pickle.load(open('stock.pkl', "rb"))
# Initializing the Flask object
app = Flask(__name__)

# The route for home page
@app.route("/")
def main():
    return render_template("home.html")


# Taking the user input and making predictions 

@app.route("/predict", methods=["POST"])

def home():
    data1 = float(request.form["a"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    data5 = float(request.form["e"])

    arr = np.array([[data1, data2, data3, data4, data5]])

    pred = model.predict(arr)
    return render_template("after.html", data=np.round(pred, 2))

if __name__ == "__main__":
    app.run(debug=True)
