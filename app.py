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
        examples = [
    # Positive Examples
    cohere.ClassifyExample(text="The order came 5 days early", label="positive"),
    cohere.ClassifyExample(text="The item exceeded my expectations", label="positive"),
    cohere.ClassifyExample(text="I ordered more for my friends", label="positive"),
    cohere.ClassifyExample(text="I would buy this again", label="positive"),
    cohere.ClassifyExample(text="I would recommend this to others", label="positive"),
    cohere.ClassifyExample(text="Rachel Burns' fundraiser has raised £48,000 towards its £60,000 target", label="positive"),
    cohere.ClassifyExample(text="The experimental treatment could give Rachel more time with her daughter", label="positive"),
    cohere.ClassifyExample(text="The money raised is being used to fund treatment in Germany", label="positive"),
    cohere.ClassifyExample(text="Rachel and her partner are hopeful about the treatment", label="positive"),
    cohere.ClassifyExample(text="The community support has been overwhelming", label="positive"),

    # Negative Examples
    cohere.ClassifyExample(text="The package was damaged", label="negative"),
    cohere.ClassifyExample(text="The order is 5 days late", label="negative"),
    cohere.ClassifyExample(text="The order was incorrect", label="negative"),
    cohere.ClassifyExample(text="I want to return my item", label="negative"),
    cohere.ClassifyExample(text="The item's material feels low quality", label="negative"),
    cohere.ClassifyExample(text="Rachel has been diagnosed with an advanced-stage brain tumour", label="negative"),
    cohere.ClassifyExample(text="She has been given four months to live", label="negative"),
    cohere.ClassifyExample(text="The treatment is very expensive", label="negative"),
    cohere.ClassifyExample(text="There are no guarantees that the treatment will work", label="negative"),
    cohere.ClassifyExample(text="The diagnosis has been devastating for her family", label="negative"),

    # Neutral Examples
    cohere.ClassifyExample(text="The product was okay", label="neutral"),
    cohere.ClassifyExample(text="I received five items in total", label="neutral"),
    cohere.ClassifyExample(text="I bought it from the website", label="neutral"),
    cohere.ClassifyExample(text="I used the product this morning", label="neutral"),
    cohere.ClassifyExample(text="The product arrived yesterday", label="neutral"),
    cohere.ClassifyExample(text="Rachel is a 22-year-old from Northern Ireland", label="neutral"),
    cohere.ClassifyExample(text="She and her partner have set up a GoFundMe page", label="neutral"),
    cohere.ClassifyExample(text="Her daughter Raeya turned one last month", label="neutral"),
    cohere.ClassifyExample(text="The fundraiser is for an experimental treatment called ONC201", label="neutral"),
    cohere.ClassifyExample(text="The fundraiser's target is £60,000", label="neutral")
]
    )
    sentiment = sentiment_response.classifications[0].prediction

    return render_template('index.html', summary=summary, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
