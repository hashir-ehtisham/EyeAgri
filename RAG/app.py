import gradio as gr
import os
import pandas as pd
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss

try:
    with open("eyeagri-dataset.txt", "r", encoding="utf-8") as f:
        agri_data = f.read()
except FileNotFoundError:
    agri_data = "EyeAgri knowledge base active. How can I help with your crops today?"

agri_chunks = [c.strip() for c in agri_data.split("\n\n") if c.strip()]
agri_embedder = SentenceTransformer('all-MiniLM-L6-v2')
agri_embeddings = agri_embedder.encode(agri_chunks)

agri_dimension = agri_embeddings.shape[1]
agri_index = faiss.IndexFlatL2(agri_dimension)
agri_index.add(agri_embeddings)

csv_context_store = {"text": ""}

# Core Logic
def eyeagri_retrieve(query, top_k=3):
    if isinstance(query, dict):
        query = query.get("content", "")
    
    query_emb = agri_embedder.encode([str(query)])
    distances, indices = agri_index.search(query_emb, top_k)
    retrieved_chunks = [agri_chunks[i] for i in indices[0] if i < len(agri_chunks)]
    return "\n".join(retrieved_chunks)

def eyeagri_respond(
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    if not history or not hf_token:
        yield history
        return

    # Extract the string content from the last message dictionary
    last_message = history[-1]
    user_query = last_message["content"] if isinstance(last_message, dict) else last_message[0]
    
    retrieved_context = eyeagri_retrieve(user_query)
    eyeagri_augmented_system = (
        f"{system_message}\n\n"
        "Relevant agricultural context:\n"
        f"{retrieved_context}\n\n"
        "Provide expert advice based on the context above."
    )

    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")
    
    messages = [{"role": "system", "content": eyeagri_augmented_system}]
    messages.extend(history)

    history.append({"role": "assistant", "content": ""})
    
    response_content = ""
    try:
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = chunk.choices[0].delta.content or ""
            response_content += token
            history[-1]["content"] = response_content
            yield history
    except Exception as e:
        history[-1]["content"] = f"Error: {str(e)}"
        yield history

# --- CSV Helpers ---
def handle_csv_upload(file):
    if file is None:
        return None, "", gr.update(interactive=False)
    try:
        df = pd.read_csv(file.name)
        summary = f"Data Summary:\nRows: {len(df)}\nColumns: {', '.join(df.columns)}\n\n"
        summary += df.head(5).to_string()
        csv_context_store["text"] = summary
        return df, f"✅ Loaded {os.path.basename(file.name)}", gr.update(interactive=True)
    except Exception as e:
        return None, f"❌ Error: {str(e)}", gr.update(interactive=False)

def copy_csv_to_prompt():
    text = csv_context_store.get("text", "")
    if not text: return "", "⚠️ No CSV."
    return f"Analyze this agricultural data:\n\n{text}", "📋 Data copied! Switch to Chat tab."

# --- UI Layout ---
with gr.Blocks(title="EyeAgri AI") as eyeagri_demo:
    with gr.Sidebar():
        gr.LoginButton()
        system_msg = gr.Textbox(value="You are EyeAgri, an AI agriculture expert, your job is to analyze the logged data and provide report and recommendations of whole data not just few rows. Also you are not allowed to answer any prompts other than agriculture.", label="System message", visible = False)
        max_toks = gr.Slider(1, 4096, 2048, label="Max Tokens", visible = False)
        temp = gr.Slider(0.1, 2.0, 0.7, label="Temperature", visible = False)
        top_p_val = gr.Slider(0.1, 1.0, 0.95, label="Top-p", visible = False)

    with gr.Tabs():
        with gr.Tab("📊 Upload Data"):
            csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
            csv_status = gr.Markdown("Ready.")
            copy_btn = gr.Button("📋 Copy Data to Chat", variant="primary", interactive=False)
            csv_preview = gr.Dataframe(label="Preview")
            copy_status = gr.Markdown("")

        with gr.Tab("🌾 Chat with EyeAgri"):
            chatbot = gr.Chatbot(label="EyeAgri Assistant", height=500)
            msg_input = gr.Textbox(placeholder="Ask about crops, soil, or pests...", label="Your Input", lines=2)
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    shared_prompt = gr.Textbox(visible=False)

    csv_file.change(handle_csv_upload, [csv_file], [csv_preview, csv_status, copy_btn])
    copy_btn.click(copy_csv_to_prompt, [], [shared_prompt, copy_status])
    shared_prompt.change(lambda x: x, [shared_prompt], [msg_input])

    def user_step(user_msg, chat_history):
        if not user_msg: return "", chat_history
        return "", chat_history + [{"role": "user", "content": user_msg}]

    triggers = [submit_btn.click, msg_input.submit]
    for trigger in triggers:
        trigger(
            fn=user_step,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
            queue=False
        ).then(
            fn=eyeagri_respond,
            inputs=[chatbot, system_msg, max_toks, temp, top_p_val],
            outputs=[chatbot]
        )

    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    eyeagri_demo.launch()
