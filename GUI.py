import tkinter as tk
from tkinter import scrolledtext

# Function to handle sending messages
def send_message():
    user_message = user_input.get()
    if user_message.strip():
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "You: " + user_message + "\n")
        user_input.delete(0, tk.END)

        # Bot response (can be modified with chatbot logic)
        bot_response = "Bot: I’m here to help!"
        chat_log.insert(tk.END, bot_response + "\n")
        chat_log.config(state=tk.DISABLED)
        chat_log.yview(tk.END)  # Auto-scroll to the latest message

# Function to clear the chat log
def clear_chat():
    chat_log.config(state=tk.NORMAL)
    chat_log.delete(1.0, tk.END)
    chat_log.config(state=tk.DISABLED)

# Function to close the window
def close_window():
    window.quit()

# Create the main window
window = tk.Tk()
window.title("WitChat Bot")
window.geometry("400x500")

# Chat log (scrollable text area)
chat_log = scrolledtext.ScrolledText(window, state=tk.DISABLED, wrap=tk.WORD, width=50, height=20)
chat_log.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Label above user input
label = tk.Label(window, text="What's your question?")
label.grid(row=1, column=0, columnspan=2)

# User input field
user_input = tk.Entry(window, width=40)
user_input.grid(row=2, column=0, padx=10, pady=10)

# Send button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.grid(row=2, column=1, padx=10, pady=10)

# Clear chat button
clear_button = tk.Button(window, text="Clear Chat", command=clear_chat)
clear_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")

# Exit button
exit_button = tk.Button(window, text="Exit", command=close_window)
exit_button.grid(row=3, column=1, padx=10, pady=10, sticky="e")

# Run the Tkinter event loop
window.mainloop()

