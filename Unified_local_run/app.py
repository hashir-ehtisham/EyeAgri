import gradio as gr
import os
import pandas as pd
import tempfile
from datetime import date, datetime
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss


# LOCAL CONFIG — adjust paths here if needed

MODEL_PATH   = r"" #path to yolo's pytorch file
DATASET_PATH = r"" #path to RAG dataset
HF_TOKEN     = "" #enter HF token here


# YOLO Model

model = YOLO(MODEL_PATH)


# RAG Setup

try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        agri_data = f.read()
except FileNotFoundError:
    agri_data = "EyeAgri knowledge base active. How can I help with your crops today?"

agri_chunks = [c.strip() for c in agri_data.split("\n\n") if c.strip()]
agri_embedder = SentenceTransformer('all-MiniLM-L6-v2')
agri_embeddings = agri_embedder.encode(agri_chunks)
agri_index = faiss.IndexFlatL2(agri_embeddings.shape[1])
agri_index.add(agri_embeddings)


# Global State

farm_state = {
    "total_area": None,
    "unit": None,
    "num_patches": None,
    "patch_size": None,
    "df": None,
}
csv_context_store = {"text": ""}
yolo_loaded_df = None
yolo_modified_csv_path = None


# Helpers

UNIT_CONVERSION = {"Acres": 4840, "Marlas": 30.25, "Sq Ft": 1}

LOG_COLUMNS = [
    "Log Date", "Total Crop Size (yards)", "Patch ID", "Patch Size (yards)",
    "Crop Type", "Number of Patches", "Crop Start Date", "Crop Duration",
    "Water Usage (L per patch)", "Irrigation Method", "Water Frequency",
    "Fertilizer Type", "Fertilizer Quantity (kg)", "NPK Ratio",
    "Application Method", "Application Timing", "Soil Moisture Level (%)",
    "Soil Type", "Soil pH Level", "Soil Temperature (C)", "Drainage Condition",
    "Air Temperature (C)", "Humidity (%)", "Rainfall", "Rainfall Duration",
    "Sunlight (Hours)", "Wind Conditions", "Plant Height (cm)",
    "Leaf Color", "Health Status"
]

def toggle_first_time(is_first):
    show_setup = is_first == "Yes"
    show_csv   = not show_setup
    return (
        gr.update(visible=show_setup),
        gr.update(visible=show_csv),
    )

def init_farm(total_area, unit, num_patches):
    if not total_area or total_area <= 0:
        return "❌ Enter a valid total area.", gr.update()
    if not num_patches or num_patches < 1:
        return "❌ Enter a valid number of patches.", gr.update()

    sq_yards   = total_area * UNIT_CONVERSION.get(unit, 1)
    patch_size = sq_yards / num_patches

    farm_state["total_area"]  = total_area
    farm_state["unit"]        = unit
    farm_state["num_patches"] = int(num_patches)
    farm_state["patch_size"]  = patch_size
    farm_state["df"]          = pd.DataFrame(columns=LOG_COLUMNS)

    patch_ids = [f"P{i+1}" for i in range(int(num_patches))]
    msg = (
        f"✅ Farm configured!\n"
        f"  Total area : {total_area} {unit} ({sq_yards:.1f} sq yards)\n"
        f"  Patches    : {int(num_patches)}\n"
        f"  Patch size : {patch_size:.2f} sq yards each"
    )
    return msg, gr.update(choices=patch_ids, value=patch_ids[0])

def load_previous_csv(file):
    if file is None:
        return "⚠️ No file uploaded.", gr.update()
    try:
        df = pd.read_csv(file.name)
        df.columns = df.columns.str.strip()
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        farm_state["df"] = df

        if not df.empty:
            row = df.iloc[0]
            try:
                farm_state["total_area"]  = float(str(row.get("Total Crop Size (yards)", 0)).replace(",", "") or 0)
                farm_state["num_patches"] = int(row.get("Number of Patches", 1) or 1)
                farm_state["patch_size"]  = float(str(row.get("Patch Size (yards)", 0)).replace(",", "") or 0)
            except Exception:
                pass

        patches = sorted(df["Patch ID"].dropna().unique().tolist()) if "Patch ID" in df.columns else []
        msg = f"✅ Previous CSV loaded! {len(df)} existing rows found."
        return msg, gr.update(choices=patches, value=patches[0] if patches else None)
    except Exception as e:
        return f"❌ Error: {e}", gr.update()

def add_log_entry(
    patch_id, crop_type, log_date, crop_start_date, crop_duration,
    water_usage, irrigation_method, water_frequency,
    fertilizer_type, fertilizer_qty, npk_ratio, app_method, app_timing,
    soil_moisture, soil_type, soil_ph, soil_temp, drainage,
    air_temp, humidity, rainfall, rainfall_duration, sunlight,
    wind_conditions, plant_height, leaf_color
):
    if farm_state["df"] is None:
        return "❌ Please configure your farm first (Step 1).", None, None

    total_sq = (farm_state["total_area"] or 0) * UNIT_CONVERSION.get(farm_state["unit"] or "Sq Ft", 1)

    new_row = {
        "Log Date": str(log_date),
        "Total Crop Size (yards)": round(total_sq, 2),
        "Patch ID": patch_id,
        "Patch Size (yards)": round(farm_state["patch_size"] or 0, 2),
        "Crop Type": crop_type,
        "Number of Patches": farm_state["num_patches"] or "",
        "Crop Start Date": str(crop_start_date),
        "Crop Duration": crop_duration,
        "Water Usage (L per patch)": water_usage,
        "Irrigation Method": irrigation_method,
        "Water Frequency": water_frequency,
        "Fertilizer Type": fertilizer_type,
        "Fertilizer Quantity (kg)": fertilizer_qty,
        "NPK Ratio": npk_ratio,
        "Application Method": app_method,
        "Application Timing": app_timing,
        "Soil Moisture Level (%)": soil_moisture,
        "Soil Type": soil_type,
        "Soil pH Level": soil_ph,
        "Soil Temperature (C)": soil_temp,
        "Drainage Condition": drainage,
        "Air Temperature (C)": air_temp,
        "Humidity (%)": humidity,
        "Rainfall": rainfall,
        "Rainfall Duration": rainfall_duration,
        "Sunlight (Hours)": sunlight,
        "Wind Conditions": wind_conditions,
        "Plant Height (cm)": plant_height,
        "Leaf Color": leaf_color,
        "Health Status": "Healthy",
    }

    farm_state["df"] = pd.concat(
        [farm_state["df"], pd.DataFrame([new_row])], ignore_index=True
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="eyeagri_log_")
    farm_state["df"].to_csv(tmp.name, index=False)

    status = f"✅ Entry logged for Patch {patch_id} on {log_date}. Total rows: {len(farm_state['df'])}"
    return status, farm_state["df"].tail(5), tmp.name

def download_log_csv():
    if farm_state["df"] is None or farm_state["df"].empty:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="eyeagri_log_")
    farm_state["df"].to_csv(tmp.name, index=False)
    return tmp.name


# YOLO functions

def yolo_load_csv(csv_file):
    global yolo_loaded_df
    if csv_file is None:
        return "⚠️ No file uploaded.", gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    try:
        df = pd.read_csv(csv_file.name)
        df.columns = df.columns.str.strip()
        required_cols = ['Log Date', 'Patch ID', 'Health Status']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"❌ Missing columns: {missing}", gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        yolo_loaded_df = df
        dates   = sorted(df['Log Date'].dropna().unique().tolist())
        patches = sorted(df['Patch ID'].dropna().unique().tolist())
        preview = f"✅ CSV loaded! Rows: {len(df)} | Dates: {len(dates)} | Patches: {len(patches)}"
        return preview, gr.update(choices=dates, value=dates[0] if dates else None), gr.update(choices=patches, value=patches[0] if patches else None)
    except Exception as e:
        return f"❌ Error: {e}", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

def yolo_analyze(img, selected_date, selected_patch, csv_file):
    global yolo_loaded_df, yolo_modified_csv_path
    if img is None:            return None, "⚠️ Upload an image.", None, gr.update(visible=False)
    if yolo_loaded_df is None: return None, "⚠️ Upload a CSV first.", None, gr.update(visible=False)
    if not selected_date:      return None, "⚠️ Select a date.", None, gr.update(visible=False)
    if not selected_patch:     return None, "⚠️ Select a patch.", None, gr.update(visible=False)

    results = model.predict(source=img, conf=0.25)
    res_plotted = results[0].plot()
    output_image = Image.fromarray(res_plotted[..., ::-1])

    names     = model.names
    probs     = results[0].probs.data.tolist()
    top_idx   = int(results[0].probs.top1)
    top_class = names[top_idx]
    top_conf  = probs[top_idx]
    confidences = {names[i]: probs[i] for i in range(len(names))}

    top_lower = top_class.lower()
    if "healthy" in top_lower:
        disease_label = None
        yolo_summary  = f"🌿 YOLO: Healthy ({top_conf:.1%} confidence)\nNo CSV update required."
    elif "brown" in top_lower:
        disease_label = "Diseased: Brown spot"
        yolo_summary  = f"🟤 YOLO: Brown Spot ({top_conf:.1%} confidence)"
    elif "bacterial" in top_lower or "blight" in top_lower:
        disease_label = "Diseased: Bacterial leaf blight"
        yolo_summary  = f"🔴 YOLO: Bacterial Leaf Blight ({top_conf:.1%} confidence)"
    else:
        disease_label = f"Diseased: {top_class}"
        yolo_summary  = f"⚠️ YOLO: {top_class} ({top_conf:.1%} confidence)"

    df = yolo_loaded_df.copy()
    df['Log Date'] = df['Log Date'].astype(str).str.strip()
    mask = (df['Log Date'] == str(selected_date).strip()) & \
           (df['Patch ID'].astype(str).str.strip() == str(selected_patch).strip())

    if df[mask].empty:
        return output_image, confidences, f"{yolo_summary}\n\n⚠️ No matching row found. CSV not modified.", None, gr.update(visible=False)

    if disease_label:
        old_vals = df.loc[mask, 'Health Status'].tolist()
        df.loc[mask, 'Health Status'] = disease_label
        yolo_loaded_df = df
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="updated_crop_log_")
        df.to_csv(tmp.name, index=False)
        yolo_modified_csv_path = tmp.name
        status = f"{yolo_summary}\n\n✅ CSV Updated! {old_vals} → '{disease_label}' ({len(df[mask])} row(s))"
        return output_image, confidences, status, gr.update(visible=True, value=tmp.name)
    else:
        status = f"{yolo_summary}\n\nℹ️ Healthy — no changes made."
        return output_image, confidences, status, gr.update(visible=False)

def yolo_get_final_csv():
    global yolo_loaded_df
    if yolo_loaded_df is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="final_crop_log_")
    yolo_loaded_df.to_csv(tmp.name, index=False)
    return tmp.name

# Chat / RAG functions

def eyeagri_retrieve(query, top_k=3):
    if isinstance(query, dict):
        query = query.get("content", "")
    query_emb = agri_embedder.encode([str(query)])
    distances, indices = agri_index.search(query_emb, top_k)
    return "\n".join([agri_chunks[i] for i in indices[0] if i < len(agri_chunks)])

def eyeagri_respond(history, system_message, max_tokens, temperature, top_p):
    """
    Local version — uses hardcoded HF_TOKEN.
    history is a list of {"role": ..., "content": ...} dicts (Gradio 6 messages format).
    """
    if not history:
        yield history
        return

    # Last entry is {"role": "user", "content": ...} added by user_step
    user_query = history[-1]["content"]

    retrieved_context = eyeagri_retrieve(user_query)
    augmented_system = (
        f"{system_message}\n\nRelevant agricultural context:\n{retrieved_context}\n\n"
        "Provide expert advice based on the context above."
    )

    # Build API messages from full history
    api_messages = [{"role": "system", "content": augmented_system}]
    for msg in history:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Append empty assistant message for streaming
    history = history + [{"role": "assistant", "content": ""}]

    client = InferenceClient(token=HF_TOKEN, model="openai/gpt-oss-20b")
    try:
        response_text = ""
        for chunk in client.chat_completion(
            api_messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p
        ):
            token = chunk.choices[0].delta.content or ""
            response_text += token
            history[-1] = {"role": "assistant", "content": response_text}
            yield history
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"Error: {e}"}
        yield history

def handle_csv_upload_chat(file):
    if file is None:
        return None, "", gr.update(interactive=False)
    try:
        df = pd.read_csv(file.name)
        summary = f"Data Summary:\nRows: {len(df)}\nColumns: {', '.join(df.columns)}\n\n" + df.head(5).to_string()
        csv_context_store["text"] = summary
        return df, f"✅ Loaded {os.path.basename(file.name)}", gr.update(interactive=True)
    except Exception as e:
        return None, f"❌ Error: {e}", gr.update(interactive=False)

def copy_csv_to_prompt():
    text = csv_context_store.get("text", "")
    if not text:
        return "", "⚠️ No CSV loaded."
    return f"Analyze this agricultural data:\n\n{text}", "📋 Data copied! Switch to Chat tab."


# Theme

green_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.gray,
).set(
    button_primary_background_fill="#16a34a",
    button_primary_background_fill_hover="#15803d",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#bbf7d0",
    button_secondary_background_fill_hover="#86efac",
    button_secondary_text_color="#14532d",
    block_label_text_color="#15803d",
    block_title_text_color="#14532d",
)


# UI
with gr.Blocks(title="EyeAgri Unified") as demo:

    gr.Markdown("# 🌾 EyeAgri — Unified Farm Intelligence Platform")
    gr.Markdown("**Developed by:** Hashir Ehtisham")

    with gr.Tabs():

        
        # TAB 1 — DATA LOGGING
        with gr.Tab("📋 Data Logging"):
            gr.Markdown("## Step 1 — Farm Setup")

            first_time_radio = gr.Radio(
                choices=["Yes", "No"],
                value="Yes",
                label="Is this your first time logging data?",
            )

            with gr.Group(visible=True) as farm_setup_group:
                gr.Markdown("### 🏡 Enter Farm Details")
                with gr.Row():
                    total_area_input  = gr.Number(label="Total Farm Area", minimum=0.01, precision=2)
                    unit_dropdown     = gr.Dropdown(choices=["Acres", "Marlas", "Sq Ft"], value="Acres", label="Unit")
                    num_patches_input = gr.Number(label="Number of Patches", minimum=1, precision=0)
                init_btn    = gr.Button("⚙️ Configure Farm", variant="primary")
                farm_status = gr.Textbox(label="Farm Status", interactive=False, lines=3)

            with gr.Group(visible=False) as prev_csv_group:
                gr.Markdown("### 📂 Load Previous Log CSV")
                prev_csv_input  = gr.File(label="Upload Previous Log CSV", file_types=[".csv"])
                prev_csv_status = gr.Textbox(label="Load Status", interactive=False, lines=2)

            gr.Markdown("---")
            gr.Markdown("## Step 2 — Log an Entry")

            with gr.Row():
                patch_id_dropdown = gr.Dropdown(label="🌾 Patch ID", choices=[], interactive=True)
                log_date_input    = gr.Textbox(label="📅 Log Date", value=str(date.today()), placeholder="YYYY-MM-DD")
                crop_type_input   = gr.Textbox(label="🌱 Crop Type", placeholder="e.g. Rice, Wheat")

            with gr.Row():
                crop_start_date = gr.Textbox(label="Crop Start Date", placeholder="YYYY-MM-DD")
                crop_duration   = gr.Textbox(label="Crop Duration", placeholder="e.g. 120 days")

            gr.Markdown("#### 💧 Water & Irrigation")
            with gr.Row():
                water_usage       = gr.Number(label="Water Usage (L per patch)")
                irrigation_method = gr.Dropdown(choices=["Drip", "Sprinkler", "Flood", "Furrow", "Manual"], label="Irrigation Method")
                water_frequency   = gr.Textbox(label="Water Frequency", placeholder="e.g. Daily, Every 3 days")

            gr.Markdown("#### 🧪 Fertilizer")
            with gr.Row():
                fertilizer_type = gr.Textbox(label="Fertilizer Type", placeholder="e.g. Urea, DAP")
                fertilizer_qty  = gr.Number(label="Fertilizer Quantity (kg)")
                npk_ratio       = gr.Textbox(label="NPK Ratio", placeholder="e.g. 10-26-26")
            with gr.Row():
                app_method = gr.Dropdown(choices=["Broadcasting", "Banding", "Foliar Spray", "Fertigation"], label="Application Method")
                app_timing = gr.Textbox(label="Application Timing", placeholder="e.g. Pre-sowing, Post-emergence")

            gr.Markdown("#### 🌍 Soil")
            with gr.Row():
                soil_moisture = gr.Number(label="Soil Moisture (%)", minimum=0, maximum=100)
                soil_type     = gr.Dropdown(choices=["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky"], label="Soil Type")
                soil_ph       = gr.Number(label="Soil pH Level", minimum=0, maximum=14, precision=1)
            with gr.Row():
                soil_temp = gr.Number(label="Soil Temperature (°C)")
                drainage  = gr.Dropdown(choices=["Well-drained", "Poorly-drained", "Moderately-drained", "Waterlogged"], label="Drainage Condition")

            gr.Markdown("#### ☀️ Weather")
            with gr.Row():
                air_temp = gr.Number(label="Air Temperature (°C)")
                humidity = gr.Number(label="Humidity (%)", minimum=0, maximum=100)
                sunlight = gr.Number(label="Sunlight (Hours)", minimum=0, maximum=24)
            with gr.Row():
                rainfall          = gr.Dropdown(choices=["None", "Light", "Moderate", "Heavy"], label="Rainfall")
                rainfall_duration = gr.Textbox(label="Rainfall Duration", placeholder="e.g. 2 hours")
                wind_conditions   = gr.Dropdown(choices=["Calm", "Light breeze", "Moderate wind", "Strong wind"], label="Wind Conditions")

            gr.Markdown("#### 🌿 Plant Observation")
            with gr.Row():
                plant_height = gr.Number(label="Plant Height (cm)")
                leaf_color   = gr.Dropdown(choices=["Dark Green", "Light Green", "Yellow", "Brown", "Mixed"], label="Leaf Color")

            log_btn     = gr.Button("📝 Add Log Entry", variant="primary")
            log_status  = gr.Textbox(label="Log Status", interactive=False, lines=2)
            log_preview = gr.Dataframe(label="Recent Entries (last 5 rows)", interactive=False)

            gr.Markdown("---")
            gr.Markdown("## Step 3 — Download Log")
            download_log_btn  = gr.Button("⬇️ Download Log CSV", variant="secondary")
            download_log_file = gr.File(label="Log CSV", interactive=False)

            # Event wiring
            first_time_radio.change(
                fn=toggle_first_time,
                inputs=first_time_radio,
                outputs=[farm_setup_group, prev_csv_group]
            )
            init_btn.click(
                fn=init_farm,
                inputs=[total_area_input, unit_dropdown, num_patches_input],
                outputs=[farm_status, patch_id_dropdown]
            )
            prev_csv_input.change(
                fn=load_previous_csv,
                inputs=prev_csv_input,
                outputs=[prev_csv_status, patch_id_dropdown]
            )
            log_btn.click(
                fn=add_log_entry,
                inputs=[
                    patch_id_dropdown, crop_type_input, log_date_input,
                    crop_start_date, crop_duration, water_usage, irrigation_method,
                    water_frequency, fertilizer_type, fertilizer_qty, npk_ratio,
                    app_method, app_timing, soil_moisture, soil_type, soil_ph,
                    soil_temp, drainage, air_temp, humidity, rainfall,
                    rainfall_duration, sunlight, wind_conditions,
                    plant_height, leaf_color
                ],
                outputs=[log_status, log_preview, download_log_file]
            )
            download_log_btn.click(
                fn=download_log_csv,
                inputs=[],
                outputs=[download_log_file]
            )


        # TAB 2 — YOLO ANALYSIS
        with gr.Tab("🔬 YOLO Analysis"):
            gr.Markdown("## YOLOv11 (nano) — Crop Disease Classification")

            with gr.Group():
                gr.Markdown("### 📂 Step 1 — Upload Crop Log CSV")
                yolo_csv_input      = gr.File(label="Upload CSV File", file_types=[".csv"])
                yolo_csv_status     = gr.Textbox(label="CSV Status", interactive=False, lines=3)
                yolo_date_dropdown  = gr.Dropdown(label="📅 Select Date",     choices=[], interactive=True)
                yolo_patch_dropdown = gr.Dropdown(label="🌾 Select Patch ID", choices=[], interactive=True)
                yolo_csv_input.change(
                    fn=yolo_load_csv,
                    inputs=yolo_csv_input,
                    outputs=[yolo_csv_status, yolo_date_dropdown, yolo_patch_dropdown]
                )

            gr.Markdown("### 🔬 Step 2 — Upload Crop Image & Analyze")
            with gr.Row():
                with gr.Column():
                    yolo_input_img   = gr.Image(type="pil", label="Input Crop Image")
                    yolo_analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                with gr.Column():
                    yolo_output_img   = gr.Image(type="pil", label="Analysis Result")
                    yolo_output_label = gr.Label(label="Confidence Scores")

            gr.Markdown("### 📊 Step 3 — Results & CSV Update")
            yolo_analysis_status = gr.Textbox(label="Analysis & CSV Update Status", interactive=False, lines=6)
            yolo_auto_download   = gr.File(label="⬇️ Updated CSV (this analysis)", visible=False, interactive=False)

            gr.Markdown("### 💾 Step 4 — Download Final CSV")
            yolo_download_btn   = gr.Button("⬇️ Download Final Updated CSV", variant="secondary")
            yolo_final_csv_file = gr.File(label="Final CSV File", interactive=False)

            yolo_analyze_btn.click(
                fn=yolo_analyze,
                inputs=[yolo_input_img, yolo_date_dropdown, yolo_patch_dropdown, yolo_csv_input],
                outputs=[yolo_output_img, yolo_output_label, yolo_analysis_status, yolo_auto_download]
            )
            yolo_download_btn.click(fn=yolo_get_final_csv, inputs=[], outputs=[yolo_final_csv_file])

        # TAB 3 — UPLOAD DATA (for Chat)
        with gr.Tab("📊 Upload Data"):
            gr.Markdown("## Upload Final Data to 🌾 EyeAgri Chat")

            chat_csv_file    = gr.File(label="Upload CSV", file_types=[".csv"])
            chat_csv_status  = gr.Markdown("Ready.")
            copy_btn         = gr.Button("📋 Copy Data to Chat", variant="primary", interactive=False)
            chat_csv_preview = gr.Dataframe(label="Preview")
            copy_status      = gr.Markdown("")

            chat_csv_file.change(
                handle_csv_upload_chat,
                [chat_csv_file],
                [chat_csv_preview, chat_csv_status, copy_btn]
            )

        # TAB 4 — CHAT
        with gr.Tab("🌾 Chat with EyeAgri"):
            # No login button needed locally — token is hardcoded above
            gr.Markdown("## 🌾 EyeAgri Chat")

            # Hidden sliders (keep defaults, advanced users can expose them)
            system_msg = gr.Textbox(
                value="You are EyeAgri, an AI agriculture expert. Analyze logged data and provide comprehensive reports and recommendations. Do not answer non-agriculture questions. You may discuss the authors or tech stack if asked.",
                label="System message",
                visible=False
            )
            max_toks  = gr.Slider(1, 4096, 2048, label="Max Tokens",   visible=False)
            temp      = gr.Slider(0.1, 2.0, 0.7,  label="Temperature", visible=False)
            top_p_val = gr.Slider(0.1, 1.0, 0.95, label="Top-p",       visible=False)

            chatbot   = gr.Chatbot(label="EyeAgri Assistant", height=500)
            msg_input = gr.Textbox(
                placeholder="Ask about crops, soil, pests, or paste your data...",
                label="Your Input",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn  = gr.Button("Clear Chat")

    # shared hidden state between Upload tab and Chat tab
    shared_prompt = gr.Textbox(visible=False)

    copy_btn.click(copy_csv_to_prompt, [], [shared_prompt, copy_status])
    shared_prompt.change(lambda x: x, [shared_prompt], [msg_input])

    def user_step(user_msg, chat_history):
        if not user_msg:
            return "", chat_history
        return "", chat_history + [{"role": "user", "content": user_msg}]

    for trigger in [submit_btn.click, msg_input.submit]:
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
    demo.launch(theme=green_theme)
