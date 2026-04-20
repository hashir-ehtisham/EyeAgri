import gradio as gr
import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss

# Load text from stored data
with open("eyeagri-dataset.txt", "r", encoding="utf-8") as f:
    agri_data = f.read()

# Simple chunking
agri_chunks = agri_data.split("\n\n")

agri_embedder = SentenceTransformer('all-MiniLM-L6-v2')
agri_embeddings = agri_embedder.encode(agri_chunks)

# Create Faiss index
agri_dimension = agri_embeddings.shape[1]
agri_index = faiss.IndexFlatL2(agri_dimension)
agri_index.add(agri_embeddings)

def eyeagri_retrieve(query, top_k=3):
    query_emb = agri_embedder.encode([query])
    distances, indices = agri_index.search(query_emb, top_k)
    retrieved_chunks = [agri_chunks[i] for i in indices[0]]
    return "\n".join(retrieved_chunks)

def eyeagri_respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    retrieved_context = eyeagri_retrieve(message)
    eyeagri_augmented_system = (
        f"{system_message}\n\n"
        "Relevant agricultural data, crop insights, and EyeAgri knowledge base:\n"
        f"{retrieved_context}\n\n"
        "Use this information while responding clearly and helpfully to farming-related queries."
    )

    client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

    eyeagri_messages = [{"role": "system", "content": eyeagri_augmented_system}]
    eyeagri_messages.extend(history)
    eyeagri_messages.append({"role": "user", "content": message})

    agri_response = ""
    for chunk in client.chat_completion(
        eyeagri_messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = chunk.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content
        agri_response += token
        yield agri_response

eyeagri_chatbot = gr.ChatInterface(
    eyeagri_respond,
    additional_inputs=[
        gr.Textbox(
            value="You are EyeAgri, an AI-powered Agriculture Management System designed by Muhammad Hashir Ehtisham, Muhammad Hamza Shahzad, Khawaja Saad Bin Waheed Lone and Muhammad Qasim Ummar, Computer Engineering students to National University of Science and Technology to assist farmers, agronomists, and agricultural researchers with crop management, soil health, pest control, irrigation, and yield optimization.",
            label="System message"
        ),
        gr.Slider(minimum=1, maximum=4096, value=2048, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks(title="EyeAgri - AI Agriculture Management System") as eyeagri_demo:
    with gr.Sidebar():
        gr.LoginButton()
    eyeagri_chatbot.render()

if __name__ == "__main__":
    eyeagri_demo.launch()
