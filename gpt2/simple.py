from transformers import pipeline
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Load the pre-trained sentiment analysis model
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Get user input text
input_text = input("Enter some text to classify sentiment: ")

# Classify the sentiment of the input text
result = classifier(input_text)[0]

# Print the predicted sentiment
if result['label'] == 'NEGATIVE':
    print("Negative sentiment")
else:
    print("Positive sentiment")
