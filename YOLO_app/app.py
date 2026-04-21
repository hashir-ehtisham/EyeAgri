import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
import tempfile
import shutil

# Load model from the root directory of Space
model = YOLO('eye_agri.pt')

# Global variable to hold the loaded dataframe and file path
loaded_df = None
loaded_csv_path = None
modified_csv_path = None


def load_csv(csv_file):
    """Load and validate the uploaded CSV file."""
    global loaded_df, loaded_csv_path

    if csv_file is None:
        return "⚠️ No file uploaded.", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    try:
        df = pd.read_csv(csv_file.name)

        # Normalize column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Check required columns
        required_cols = ['Log Date', 'Patch ID', 'Health Status']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"❌ Missing required columns: {missing}", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

        loaded_df = df
        loaded_csv_path = csv_file.name

        # Get unique dates and patch IDs
        dates = sorted(df['Log Date'].dropna().unique().tolist())
        patches = sorted(df['Patch ID'].dropna().unique().tolist())

        preview = f"✅ CSV loaded successfully!\n📋 Rows: {len(df)} | Columns: {len(df.columns)}\n📅 Dates: {len(dates)} | Patches: {len(patches)}"
        return preview, gr.update(choices=dates, value=dates[0] if dates else None), gr.update(choices=patches, value=patches[0] if patches else None)

    except Exception as e:
        return f"❌ Error loading CSV: {str(e)}", gr.update(choices=[], value=None), gr.update(choices=[], value=None)


def analyze_and_update(img, selected_date, selected_patch, csv_file):
    """Run YOLO analysis and update the CSV health status if diseased."""
    global loaded_df, modified_csv_path

    if img is None:
        return None, "⚠️ Please upload an image.", None, gr.update(visible=False)

    if loaded_df is None:
        return None, "⚠️ Please upload a CSV file first.", None, gr.update(visible=False)

    if not selected_date:
        return None, "⚠️ Please select a date.", None, gr.update(visible=False)

    if not selected_patch:
        return None, "⚠️ Please select a patch.", None, gr.update(visible=False)

    # YOLO Inference 
    results = model.predict(source=img, conf=0.25)

    # Visual output (BGR → RGB)
    res_plotted = results[0].plot()
    output_image = Image.fromarray(res_plotted[..., ::-1])

    # Classification probabilities
    names  = model.names
    probs  = results[0].probs.data.tolist()
    top_idx = int(results[0].probs.top1)
    top_class = names[top_idx]
    top_conf  = probs[top_idx]

    confidences = {names[i]: probs[i] for i in range(len(names))}

    # ── Determine disease label ──────────────────────────────────────
    top_lower = top_class.lower()
    if "healthy" in top_lower:
        disease_label = None          # no update needed
        yolo_summary  = f"🌿 YOLO Result: **Healthy** ({top_conf:.1%} confidence)\nNo CSV update required."
    elif "brown" in top_lower or "brown_spot" in top_lower or "brownspot" in top_lower:
        disease_label = "Diseased: Brown spot"
        yolo_summary  = f"🟤 YOLO Result: **Brown Spot** ({top_conf:.1%} confidence)"
    elif "bacterial" in top_lower or "blight" in top_lower or "leaf_blight" in top_lower:
        disease_label = "Diseased: Bacterial leaf blight"
        yolo_summary  = f"🔴 YOLO Result: **Bacterial Leaf Blight** ({top_conf:.1%} confidence)"
    else:
        # Fallback: treat anything non-healthy as generic disease
        disease_label = f"Diseased: {top_class}"
        yolo_summary  = f"⚠️ YOLO Result: **{top_class}** ({top_conf:.1%} confidence)"

    # ── Update CSV ───────────────────────────────────────────────────
    df = loaded_df.copy()

    # Normalise date column for flexible matching
    df['Log Date'] = df['Log Date'].astype(str).str.strip()
    selected_date_str = str(selected_date).strip()
    selected_patch_str = str(selected_patch).strip()

    mask = (df['Log Date'] == selected_date_str) & (df['Patch ID'].astype(str).str.strip() == selected_patch_str)
    matched_rows = df[mask]

    if matched_rows.empty:
        csv_status = (
            f"⚠️ No matching row found for Date='{selected_date_str}' and Patch='{selected_patch_str}'.\n"
            f"CSV was **not** modified."
        )
        return output_image, confidences, csv_status, gr.update(visible=False)

    csv_status_parts = [yolo_summary, ""]

    if disease_label:
        old_vals = df.loc[mask, 'Health Status'].tolist()
        df.loc[mask, 'Health Status'] = disease_label
        loaded_df = df          # update global

        # Save to a temp file so user can download
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="updated_crop_log_")
        df.to_csv(tmp.name, index=False)
        modified_csv_path = tmp.name

        csv_status_parts.append(
            f"✅ CSV Updated!\n"
            f"  📅 Date   : {selected_date_str}\n"
            f"  🌾 Patch  : {selected_patch_str}\n"
            f"  🔄 Changed: {old_vals} → '{disease_label}'\n"
            f"  ({len(matched_rows)} row(s) updated)"
        )
        show_download = True
    else:
        modified_csv_path = None
        csv_status_parts.append(
            f"ℹ️ Crop is Healthy — no changes made to the CSV.\n"
            f"  📅 Date  : {selected_date_str}\n"
            f"  🌾 Patch : {selected_patch_str}"
        )
        show_download = False

    csv_status = "\n".join(csv_status_parts)
    return output_image, confidences, csv_status, gr.update(visible=show_download, value=modified_csv_path if show_download else None)


def get_final_csv():
    """Save the current state of loaded_df to a temp file and return path for download."""
    global loaded_df, modified_csv_path

    if loaded_df is None:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="final_crop_log_")
    loaded_df.to_csv(tmp.name, index=False)
    modified_csv_path = tmp.name
    return tmp.name

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# YOLOv11 (nano) 🌾 EyeAgri Healthy vs Diseased Classification")
    gr.Markdown(
        "**Workflow:** Upload your crop log CSV → Select date & patch → Upload a crop image → Click **Analyze** → Download the updated CSV anytime."
    )

    with gr.Group():
        gr.Markdown("### 📂 Step 1 — Upload Crop Log CSV")
        csv_input      = gr.File(label="Upload CSV File", file_types=[".csv"])
        csv_status     = gr.Textbox(label="CSV Status", interactive=False, lines=3)
        date_dropdown  = gr.Dropdown(label="📅 Select Date",    choices=[], interactive=True)
        patch_dropdown = gr.Dropdown(label="🌾 Select Patch ID", choices=[], interactive=True)

        csv_input.change(
            fn=load_csv,
            inputs=csv_input,
            outputs=[csv_status, date_dropdown, patch_dropdown]
        )

    gr.Markdown("---")

    gr.Markdown("### 🔬 Step 2 — Upload Crop Image & Analyze")

    with gr.Row():
        with gr.Column():
            input_img   = gr.Image(type="pil", label="Input Crop Image")
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column():
            output_img   = gr.Image(type="pil", label="Analysis Result")
            output_label = gr.Label(label="Confidence Scores")

    gr.Markdown("---")
    gr.Markdown("### 📊 Step 3 — Results & CSV Update")

    analysis_status  = gr.Textbox(label="Analysis & CSV Update Status", interactive=False, lines=6)

    auto_download    = gr.File(label="⬇️ Updated CSV (this analysis)", visible=False, interactive=False)

    gr.Markdown("---")
    gr.Markdown("### 💾 Step 4 — Download Final CSV")
    gr.Markdown("Click below at any time to download the **full CSV** with all changes applied so far.")

    download_final_btn = gr.Button("⬇️ Download Final Updated CSV", variant="secondary")
    final_csv_file     = gr.File(label="Final CSV File", interactive=False)

    analyze_btn.click(
        fn=analyze_and_update,
        inputs=[input_img, date_dropdown, patch_dropdown, csv_input],
        outputs=[output_img, output_label, analysis_status, auto_download]
    )

    download_final_btn.click(
        fn=get_final_csv,
        inputs=[],
        outputs=[final_csv_file]
    )

if __name__ == "__main__":
    demo.launch()
