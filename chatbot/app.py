import gradio as gr
from chatbot import ChatbotAssistant
import nltk

nltk.download('punkt')
nltk.download('wordnet')

assistant = ChatbotAssistant()
assistant.parse_intents()
assistant.prepare_data()

bio_content = """
Biography of Mirzohamidullo Hoshimov
My name is Mirzohamidullo Hoshimov, and I was born on the 15th of December, 2002, in Khujand, Tajikistan. Growing up in one of the most historic and culturally rich cities of Central Asia has shaped me into someone who values knowledge, perseverance, and continuous growth. From a young age, I developed a deep curiosity about the world around me, and this curiosity gradually transformed into a passion for learning, exploring, and contributing meaningfully to society.
My academic journey has led me to South Korea, where I currently study at Endicott College of Woosong University. I am pursuing my studies in the faculty of Global Convergence Management, a program that provides me with a unique blend of business management and global perspectives. This field challenges me to think critically, develop cross-cultural competencies, and prepare myself for leadership roles in an interconnected world. Alongside this, I am also minoring in the Department of Artificial Intelligence and Big Data, which allows me to integrate my love for technology and data-driven solutions with my management studies.
Artificial Intelligence, Machine Learning, and Deep Learning are subjects that inspire me deeply. Although they are highly technical and complex fields, I have dedicated myself to learning them both academically and independently. I spend significant time building projects, working on algorithms, and understanding the theoretical foundations that shape the future of AI. Over time, I have accumulated dozens of certificates in machine learning, AI, and data-related disciplines, which not only validate my skills but also push me to continue growing in this fast-evolving field. My projects range from computer vision applications to predictive analytics models, and each of them represents my commitment to both learning and practical implementation.
Beyond academics, I am a person with diverse interests and hobbies. I find joy and discipline in sports, particularly football, wrestling, and swimming. These activities have taught me about teamwork, resilience, and the importance of physical well-being. Wrestling, in particular, has instilled in me a strong sense of mental toughness, while football has given me the ability to collaborate with others toward shared goals. Swimming provides me with balance and clarity, helping me remain focused in both academic and personal pursuits. In addition, I enjoy reading, which broadens my perspective and allows me to continuously challenge my way of thinking. Whether I am reading about philosophy, economics, or modern technology, books serve as an endless source of growth.
Programming is another passion of mine. Writing code, building applications, and solving computational problems excite me because they combine creativity with logic. Every project I work on teaches me new skills, from problem-solving to debugging, and it constantly reinforces the idea that technology has the power to transform industries and improve human lives. With programming as both a hobby and a tool, I aim to contribute to innovative solutions in AI, business, and beyond.
Looking ahead, I see myself as someone who will continue to bridge the worlds of management and technology. My dream is to apply my knowledge of AI, Big Data, and global management to real-world challenges, whether in entrepreneurship, research, or industry. I aspire to contribute to projects that bring about positive change, improve efficiency, and create opportunities for people across the globe. With my strong academic background, international education, and personal dedication, I am confident in my ability to grow into a professional who makes a meaningful impact.
In conclusion, my life so far has been defined by curiosity, learning, and a drive for excellence. From my roots in Khujand, Tajikistan, to my academic journey in South Korea, I carry with me the values of resilience, hard work, and passion. My hobbies, studies, and professional aspirations all come together to reflect who I am: a dedicated student, an ambitious learner, and a passionate individual committed to both personal and professional growth. The road ahead is full of opportunities, and I am determined to make the most of them by striving for knowledge, innovation, and impact.
"""

assistant.load_biography(text=bio_content)
assistant.train_model(batch_size=8, lr=0.001, epochs=50)

def chat_function(user_message):
    response, hints_str = assistant.process_message(user_message)
    return response, hints_str.replace('|', ' | ')

css = """
body, html, .gradio-container {
    background: url('https://raw.githubusercontent.com/Hafizulloevich/Hafizulloevich/main/photo.jpg') no-repeat center center fixed !important;
    background-size: cover !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gr-box, .wrap, .container, .gr-panel, .gr-row, .gr-block, .gr-column {
    background: rgba(255, 255, 255, 0.35) !important; 
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
}

/* Inputs */
textarea, input {
    background: transparent !important;
    font-size: 18px !important;
    color: #000 !important;
}

/* Bigger response box */
.gr-textbox textarea {
    min-height: 150px !important;
    font-size: 18px !important;
}

/* Buttons */
button {
    font-size: 18px !important;
    padding: 10px 20px !important;
    border-radius: 10px !important;
    background: rgba(0, 0, 0, 0.7) !important;
    color: white !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Mirzohamidullo's Biography Chatbot \nAsk me about my life!")
    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="E.g., What are your hobbies?", lines=3)
        submit_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")  

    with gr.Row():
        chatbot_response = gr.Textbox(label="Chatbot Response", interactive=False, lines=5)
        hints_output = gr.Textbox(label="Hints for Next Questions", interactive=False, placeholder="Suggestions will appear here!", lines=3)
    submit_btn.click(chat_function, inputs=user_input, outputs=[chatbot_response, hints_output])
    clear_btn.click(lambda: "", inputs=None, outputs=user_input)  

demo.launch(debug=True)
