from flask import Flask, render_template, request
from predictor import landscape_predictor


app = Flask(__name__)

land = landscape_predictor()

# Base endpoint to perform prediction.
@app.route('/', methods=['POST'])
def make_prediction():
    prediction = land.predict(request)
    return render_template('index.html', prediction=prediction, generated_text=None)


@app.route('/', methods=['GET'])
def load():
    return render_template('index.html', prediction=None, generated_text=None)


if __name__ == '__main__':
    app.run(debug=True)
