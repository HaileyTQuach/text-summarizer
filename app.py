from flask import Flask, request, render_template
import cohere

app = Flask(__name__)
co = cohere.Client('Xs7YUlr15i62ot0hhMRvQ2ceBLQFuh6vc6dmnD3p')  # Replace with your actual API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    response = co.summarize(
        text=text,
        model='summarize-xlarge',
        length='medium'
    )
    summary = response.summary
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
