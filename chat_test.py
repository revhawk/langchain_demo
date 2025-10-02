from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
#from langchain.schema import HumanMessage
#from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=api_key)

# Initialize a message history object
message_history = ChatMessageHistory()

print("Chatbot with memory is running! Type 'exit' to quit.")

while True:
    user_input = input("Human input: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Add user input to history
    message_history.add_user_message(user_input)

    # Get response from model using the full history
    response = chat_model.invoke(message_history.messages)

    # Add bot response to history
    message_history.add_ai_message(response.content)

    print("Bot:", response.content)


