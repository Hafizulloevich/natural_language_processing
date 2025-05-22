import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nltk.download('punkt_tab')
nltk.download('wordnet')

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChatbotAssistant:
    def __init__(self):
        self.model = None
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.intents_hints = {}  # New: Dictionary for hints per intent
        self.function_mappings = {'biography': self.get_bio_response}
        self.bio_text = None
        self.bio_sentences = []
        self.last_query = ''

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        intents_data = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up", "Hey", "greetings"],
                    "responses": ["Hello! My name is Mirzo, how can I help you?", "Hi there, I am Mirzo, how can I help?"],
                    "hints": ["Tell me about yourself", "What are your hobbies?", "Where were you born?"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day", "bye", "cao", "see ya"],
                    "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!"],
                    "hints": []  
                },
                {
                    "tag": "introduction",
                    "patterns": ["What does this bot can do?", "What can you do?", "Tell me about yourself", "Introduction", "Who are you?", "Introduce yourself"],
                    "responses": ["I am Mirzohamidullo Hoshimov, who is a basic chatbot for bio introduction. I am 22 years old and currently I am student!"],
                    "hints": ["What are your passions?", "Tell me about your education", "What do you study?"]
                },
                {
                    "tag": "birth",
                    "patterns": ["When were you born", "Where were you born", "What is your birth date", "Birthplace", "Origins"],
                    "responses": ["I was born on the 15th of December, 2002, in Khujand, Tajikistan. Growing up in one of the most historic and culturally rich cities of Central Asia has shaped me into someone who values knowledge, perseverance, and continuous growth."],
                    "hints": ["What do you study?", "What are your hobbies?", "Tell me about your passions"]
                },
                {
                    "tag": "education",
                    "patterns": ["Where do you study", "Education", "University", "What do you major in", "Field of study"],
                    "responses": ["My academic journey has led me to South Korea, where I currently study at Endicott College of Woosong University. I am pursuing my studies in the faculty of Global Convergence Management, a program that provides me with a unique blend of business management and global perspectives. Alongside this, I am also minoring in the Department of Artificial Intelligence and Big Data."],
                    "hints": ["What are your passions?", "What are your hobbies?", "Tell me about your future"]
                },
                {
                    "tag": "passions",
                    "patterns": ["What are your passions", "Interests in AI", "Machine learning", "Deep learning", "Projects", "Certificates"],
                    "responses": ["Artificial Intelligence, Machine Learning, and Deep Learning are subjects that inspire me deeply. I have dedicated myself to learning them both academically and independently, building projects from computer vision applications to predictive analytics models. I have accumulated dozens of certificates in machine learning, AI, and data-related disciplines."],
                    "hints": ["Do you like programming?", "Tell me about your education", "What are your future dreams?"]
                },
                {
                    "tag": "hobbies",
                    "patterns": ["What are your hobbies", "Sports", "Football", "Wrestling", "Swimming", "Reading"],
                    "responses": ["Beyond academics, I find joy and discipline in sports, particularly football, wrestling, and swimming. These activities have taught me about teamwork, resilience, and the importance of physical well-being. In addition, I enjoy reading, which broadens my perspective and allows me to continuously challenge my way of thinking."],
                    "hints": ["What do you enjoy reading?", "Tell me about football", "What are your passions?"]
                },
                {
                    "tag": "programming",
                    "patterns": ["Do you like programming", "What about coding", "Programming skills", "Building applications"],
                    "responses": ["Programming is another passion of mine. Writing code, building applications, and solving computational problems excite me because they combine creativity with logic. Every project I work on teaches me new skills, from problem-solving to debugging, and it constantly reinforces the idea that technology has the power to transform industries and improve human lives."],
                    "hints": ["What are your passions?", "Tell me about your projects", "What is your future in tech?"]
                },
                {
                    "tag": "future",
                    "patterns": ["Future plans", "Asppirations", "Career goals", "What do you want to do", "Dreams"],
                    "responses": ["Looking ahead, I see myself as someone who will continue to bridge the worlds of management and technology. My dream is to apply my knowledge of AI, Big Data, and global management to real-world challenges, whether in entrepreneurship, research, or industry, contributing to projects that bring about positive change, improve efficiency, and create opportunities for people across the globe."],
                    "hints": ["Tell me about your education", "What are your hobbies?", "What inspires you about AI?"]
                },
                {
                    "tag": "biography",
                    "patterns": ["Who is your creator", "Who is him?", "Biography of your creator", "Mirzohamidullo Hoshimov", "Creator portfolio or website", "Okay, tell me about him", "Tell me about your creator"],
                    "responses": [""], 
                    "hints": ["What are your passions?", "Tell me about your birth", "Where do you study?"]
                }
            ]
        }

        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']
                self.intents_hints[intent['tag']] = intent['hints'] 

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)
        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = ChatbotModel(self.X.shape[1], len(self.intents)) 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def load_biography(self, text=None):
        if text:
            self.bio_text = text
        else:
            self.bio_text = ''
        self.bio_sentences = nltk.sent_tokenize(self.bio_text) if self.bio_text else []
        print(f"Biography loaded ({len(self.bio_sentences)} sentences)")

    def get_bio_response(self):
        if not self.bio_sentences:
            return "I don't have biography information loaded."
        query_words = set(self.tokenize_and_lemmatize(self.last_query))
        best_sentence = ''
        max_overlap = 0
        for sentence in self.bio_sentences:
            sent_words = set(self.tokenize_and_lemmatize(sentence))
            overlap = len(query_words.intersection(sent_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence
        return best_sentence if best_sentence else "I couldn't find relevant information in my biography."

    def process_message(self, input_message):
        self.last_query = input_message
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        
    
        if self.function_mappings and predicted_intent in self.function_mappings:
            response = self.function_mappings[predicted_intent]()
        elif self.intents_responses[predicted_intent]:
            response = random.choice(self.intents_responses[predicted_intent])
        else:
            response = "I'm not sure how to respond to that."
        
        # Get hints for the intent
        hints = self.intents_hints.get(predicted_intent, ["What do you study?", "Tell me about your hobbies"])
        hints_str = "Suggestions: " + " | ".join(hints)
        
        return response, hints_str

if __name__ == '__main__':
    assistant = ChatbotAssistant()
    assistant.parse_intents()
    assistant.prepare_data()
    
    bio_content = """
    Biography of Mirzohamidullo Hoshimov
    My name is Mirzohamidullo Hoshimov, and I was born on the 15th of December, 2002, in Khujand, Tajikistan. Growing up in one of the most historic and culturally rich cities of Central Asia has shaped me into someone who values knowledge, perseverance, and continuous growth. From a young age, I developed a deep curiosity about the world around me, and this curiosity gradually transformed into a passion for learning, exploring, and contributing meaningfully to society.

    My academic journey has led me to South Korea, where I currently study at Endicott College of Woosong University. I am pursuing my studies in the faculty of Global Convergence Management, a program that provides me with a unique blend of business management and global perspectives. This field challenges me to think critically, develop cross-cultural competencies, and prepare myself for leadership roles in an interconnected world. Alongside this, I am also minoring in the Department of Artificial Intelligence and Big Data, which allows me to integrate the technical disciplines into business.

    Artificial Intelligence, Machine Learning, and Deep Learning are subjects that inspire me deeply. Although they are highly technical and complex fields, I have dedicated myself to learning them both academically and independently. I spend significant time building projects, working on algorithms, and understanding the theoretical foundations that shape the future of AI. Over time, I have accumulated dozens of certificates in machine learning, AI, and data-related disciplines, which not only validate my skills but also push me to continue growing in this fast-evolving field. My projects range from computer vision applications to predictive analytics models, and each of them represents my commitment to both learning and practical implementation.

    Beyond academics, I am a person with diverse interests and hobbies. I find joy and discipline in sports, particularly football, wrestling, and swimming. These activities have taught me about teamwork, resilience, and the importance of physical well-being. Wrestling, in particular, has instilled in me a strong sense of mental toughness, while football has given me the ability to collaborate with others toward shared goals. Swimming provides me with balance and clarity, helping me remain focused in both academic and personal pursuits. In addition, I enjoy reading, which broadens my perspective and allows me to continuously challenge my way of thinking. Whether I am reading about philosophy, economics, or modern technology, books serve as an endless source of growth.

    Programming is another passion of mine. Writing code, building applications, and solving computational problems excite me because they combine creativity with logic. Every project I work on teaches me new skills, from problem-solving to debugging, and it constantly reinforces the idea that technology has the power to transform industries and improve human lives. With programming as both a hobby and a tool, I aim to contribute to innovative solutions in AI, business, and beyond.

    Looking ahead, I see myself as someone who will continue to bridge the worlds of management and technology. My dream is to apply my knowledge of AI, Big Data, and global management to real-world challenges, whether in entrepreneurship, research, or industry. I aspire to contribute to projects that bring about positive change, improve efficiency, and create opportunities for people across the globe. With my strong academic background, international education, and personal dedication, I am confident in my ability to grow into a professional who makes a meaningful impact.

    In conclusion, my life so far has been defined by curiosity, learning, and a drive for excellence. From my roots in Khujand, Tajikistan, to my academic journey in South Korea, I carry with me the values of resilience, hard work, and passion. My hobbies, studies, and professional aspirations all come together to reflect who I am: a dedicated student, an ambitious learner, and a passionate individual committed to both personal and professional growth. The road ahead is full of opportunities, and I am determined to make the most of them by striving for knowledge, innovation, and impact.
    """
    
    assistant.load_biography(text=bio_content)
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)

    while True:
        message = input('Enter your message:')
        if message == '/quit':
            break
        response, hints = assistant.process_message(message)
        print(response)
        print(hints)  