from flask import Flask, request, render_template
import cohere
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(api_key)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    length = request.form['length']
    response = co.summarize(
        text=text,
        model='summarize-xlarge',
        length=length
    )
    summary = response.summary

    # Sentiment analysis
    sentiment_response = co.classify(
        model='large',
        inputs=[summary],
        examples=[
    cohere.ClassifyExample(text="The order came 5 days early", label="positive"), 
    cohere.ClassifyExample(text="The order came 5 days early", label="positive"), 
    cohere.ClassifyExample(text="The order came 5 days early", label="positive"), 
    cohere.ClassifyExample(text="The item exceeded my expectations", label="positive"), 
    cohere.ClassifyExample(text="I ordered more for my friends", label="positive"), 
    cohere.ClassifyExample(text="I would buy this again", label="positive"), 
    cohere.ClassifyExample(text="I would recommend this to others", label="positive"), 
    cohere.ClassifyExample(text="The package was damaged", label="negative"), 
    cohere.ClassifyExample(text="The order is 5 days late", label="negative"), 
    cohere.ClassifyExample(text="The order was incorrect", label="negative"), 
    cohere.ClassifyExample(text="I want to return my item", label="negative"), 
    cohere.ClassifyExample(text="The item\'s material feels low quality", label="negative"), 
    cohere.ClassifyExample(text="The product was okay", label="neutral"), 
    cohere.ClassifyExample(text="I received five items in total", label="neutral"), 
    cohere.ClassifyExample(text="I bought it from the website", label="neutral"), 
    cohere.ClassifyExample(text="I used the product this morning", label="neutral"), 
    cohere.ClassifyExample(text="The product arrived yesterday", label="neutral"),
]

    )
    sentiment = sentiment_response.classifications[0].prediction

    return render_template('index.html', summary=summary, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
