# 🧠 Text Summarization Tool

This project is a **Text Summarization Web App** built using **Hugging Face Transformers**, **Google's PEGASUS model** and **Gradio** for the frontend interface.

## ✨ Features

- Summarizes long texts using the powerful `google/pegasus-cnn_dailymail` model
- Handles large inputs by automatically chunking text
- Displays a real-time **word count**
- Copy the summary with one click
- Download summary as a `.txt` file
- Clean and modern UI using Gradio Blocks

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install transformers torch gradio
   ```

2. Run the app:
   ```bash
   python app.py
   ```

3. The app will launch in your browser at `http://127.0.0.1:7860/`.

## 🛠 Technologies Used

- Python
- Gradio
- Hugging Face Transformers
- PEGASUS (`google/pegasus-cnn_dailymail`)
- PyTorch

## 📷 UI Preview

> Paste your text in the input box and get clear, concise bullet point summaries.

## 📌 Example Use Cases

- Summarizing articles, research papers, or reports
- Extracting key points from meeting transcripts
- Assisting in content creation and note-taking

## 🙌 Credits

- Model: [PEGASUS by Google](https://huggingface.co/google/pegasus-cnn_dailymail)
- Interface: [Gradio](https://gradio.app)
- Developer: Uday

---

> 🔗 Feel free to fork this repo, improve it, and use it in your projects!
