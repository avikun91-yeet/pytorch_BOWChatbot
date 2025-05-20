# Pet Care Chatbot

This project is a simple NLP-based chatbot designed to assist users with questions about pet care. It uses a Bag-of-Words (BoW) approach and a feedforward neural network to understand and respond to user input.
The chatbot is trained on a custom dataset covering various pets, including dogs, cats, parrots, rabbits, and hamsters. Each species has intents related to feeding, playing, and general care.

The original dataset that was used while developing the bot is here: https://github.com/Rawkush/Chatbot/blob/master/intents.json

The youtube playlist followed is: 
https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg&ab_channel=PatrickLoeber

## Features
Answers pet-related queries with structured responses.
Covers specific species individually with targeted answers.
Organized by intent tags such as feeding, playing, and care.
Easy to extend with new pets or questions via the JSON file.

## Technologies Used

PyTorch
NLTK (Natural Language Toolkit)
Bag-of-Words NLP model
JSON for intent management



for the intents.json file>>> change the tags, responses and patterns to customize the chatbot

comment out the nltk.download("punkt_tab") after running once

change the json file path
