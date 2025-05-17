from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import gradio as gr
import torch

# Load PEGASUS model
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Summary function with chunking
def summarize_text(user_input):
    user_input = user_input.strip()
    if not user_input:
        return "⚠️ Please paste some text to summarize.", None

    # Tokenize and chunk if too long
    inputs = tokenizer(user_input, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    max_len = 1024

    # Chunk input tokens
    chunks = [input_ids[i:i + max_len] for i in range(0, len(input_ids), max_len)]
    summaries = []

    for chunk in chunks:
        input_batch = torch.unsqueeze(chunk, 0)
        summary_ids = model.generate(input_batch, max_length=160, min_length=40, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    full_summary = " ".join(summaries)

    # Format with bullet points
    if not full_summary.startswith("-"):
        full_summary = "- " + full_summary.replace(". ", ".\n- ")

    return full_summary, full_summary

# Word count display
def update_word_count(text):
    return f"**Word Count:** {len(text.strip().split())}"

# Build Gradio UI
with gr.Blocks(theme=gr.themes.Monochrome(), title="🧠 Text Summarization Tool") as ui:
    gr.Markdown("""
    # 🧠 NLP Text Summarization Tool
    Paste in any article, essay, or report. Get back clean, concise key points.
    """)

    with gr.Row():
        input_text = gr.Textbox(
            lines=15,
            placeholder="📋 Paste your article or content here...",
            label="Input Text"
        )
        output_text = gr.Textbox(
            lines=15,
            label="📌 Summarized Key Points"
        )

    word_counter = gr.Markdown("**Word Count:** 0")

    with gr.Row():
        summarize_btn = gr.Button("✨ Get Summary")
        copy_btn = gr.Button("📋 Copy Summary")
        download_btn = gr.DownloadButton("💾 Download Summary", file_name="summary.txt")

    # Live word count
    input_text.change(update_word_count, inputs=input_text, outputs=word_counter)

    # Summary button logic
    summarize_btn.click(
        summarize_text,
        inputs=input_text,
        outputs=[output_text, download_btn]
    )

    # Copy to clipboard
    copy_btn.click(
        None,
        None,
        None,
        js="""
        const output = document.querySelector('textarea[aria-label="📌 Summarized Key Points"]').value;
        navigator.clipboard.writeText(output).then(() => {
            alert('✅ Summary copied to clipboard!');
        });
        """
    )

    gr.Markdown("---")
    gr.Markdown("🧪 Powered by Google's PEGASUS Model | Created with 🤍 using Hugging Face + Gradio")

# Launch app
ui.launch()
