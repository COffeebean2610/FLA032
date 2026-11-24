import os
import sys
import json
import time
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to sys.path so src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Streamlit Page Config
st.set_page_config(
    page_title="AgriGuard | Crop Disease Diagnosis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Farmer-Friendly Clean Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e5631;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    .badge-healthy {
        background-color: #2e7d32;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-diseased {
        background-color: #c62828;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 12px;
        margin-top: 20px;
        font-size: 0.88rem;
        color: #5d4037;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_disease_info():
    info_path = os.path.join(os.path.dirname(__file__), "disease_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_class_names():
    metrics_path = os.path.join("results", "metrics", "class_names.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    # Fallback to scanning dataset or static list
    from src.data_loader import get_class_names
    try:
        return get_class_names("dataset/image data")
    except Exception:
        # Load from disease_info keys
        return sorted(list(load_disease_info().keys()))

@st.cache_resource
def load_trained_model():
    """
    Attempts to load best Keras model or TFLite model.
    """
    import tensorflow as tf
    keras_path = os.path.join("models", "best_model", "best_model.keras")
    if not os.path.exists(keras_path):
        keras_path = os.path.join("models", "MobileNetV2.keras")
        if not os.path.exists(keras_path):
            candidates = [os.path.join("models", f) for f in os.listdir("models") if f.endswith(".keras")]
            if candidates:
                keras_path = candidates[0]
            else:
                return None, "keras", None
                
    try:
        model = tf.keras.models.load_model(keras_path, safe_mode=False)
        return model, "keras", keras_path
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None, "error", None

def preprocess_image(pil_img, target_size=(224, 224)):
    img_rgb = pil_img.convert("RGB")
    img_resized = img_rgb.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor, np.array(img_resized)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
st.sidebar.title("AgriGuard Edge")
nav_choice = st.sidebar.radio("Navigation", ["🔍 Leaf Diagnosis & XAI", "📊 Supported Crops & Diseases", "ℹ️ About the Project"])

disease_info = load_disease_info()
class_names = load_class_names()
model, model_type, model_path = load_trained_model()

if nav_choice == "🔍 Leaf Diagnosis & XAI":
    st.markdown('<div class="main-header">Smart Crop Disease Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a plant leaf photograph to identify pathology, explore Grad-CAM visual explanations, and receive verified agronomic prevention guidelines.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Leaf Image")
        uploaded_file = st.file_uploader("Select JPEG or PNG image...", type=["jpg", "jpeg", "png"])
        
        sample_choice = st.selectbox(
            "Or test with a sample from dataset:",
            ["None", "Grape - Black Rot", "Tomato - Early Blight", "Rice - Brown Spot", "Potato - Late Blight"]
        )
        
        sample_paths = {
            "Grape - Black Rot": "dataset/image data/test/grape/black rot",
            "Tomato - Early Blight": "dataset/image data/test/tomato/early blight",
            "Rice - Brown Spot": "dataset/image data/test/Rice/BrownSpot",
            "Potato - Late Blight": "dataset/image data/test/potato/late blight"
        }
        
        selected_image = None
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
        elif sample_choice != "None":
            folder = sample_paths.get(sample_choice)
            if folder and os.path.exists(folder):
                files = os.listdir(folder)
                if files:
                    selected_image = Image.open(os.path.join(folder, files[0]))
                    
        if selected_image is not None:
            st.image(selected_image, caption="Target Foliage Image", use_container_width=True)
            predict_btn = st.button("🚀 Analyze Leaf Health", type="primary", use_container_width=True)
        else:
            predict_btn = False
            st.info("Please upload an image or choose a dataset sample to begin analysis.")
            
    with col2:
        st.subheader("2. Diagnostic Results")
        if predict_btn and selected_image is not None:
            if model is None:
                st.warning("⚠️ Model weights are currently initializing or training. Please train the model first.")
            else:
                with st.spinner("Processing image and generating explainability heatmap..."):
                    img_tensor, original_rgb = preprocess_image(selected_image)
                    t0 = time.perf_counter()
                    preds = model.predict(img_tensor, verbose=0)[0]
                    t1 = time.perf_counter()
                    latency_ms = (t1 - t0) * 1000.0
                    
                    top_indices = np.argsort(preds)[::-1][:3]
                    top_class = class_names[top_indices[0]]
                    top_conf = float(preds[top_indices[0]])
                    
                    info = disease_info.get(top_class, {
                        "crop": top_class.split("___")[0],
                        "condition": top_class.split("___")[1],
                        "status": "Healthy" if "healthy" in top_class.lower() else "Diseased",
                        "symptoms": "Detailed visual symptoms record.",
                        "prevention": "Standard cultural practices and field sanitation."
                    })
                    
                    status_badge = "badge-healthy" if info.get("status") == "Healthy" else "badge-diseased"
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>Diagnosis: <b>{info.get('crop')} - {info.get('condition')}</b></h3>
                        <p><span class="{status_badge}">{info.get('status').upper()}</span> &nbsp; Confidence: <b>{top_conf*100:.2f}%</b> &nbsp; (Latency: {latency_ms:.1f} ms)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top 3 Predictions Bar
                    st.write("**Top 3 Class Candidates:**")
                    for idx in top_indices:
                        c_name = class_names[idx].replace("___", " → ")
                        prob = float(preds[idx])
                        st.progress(prob, text=f"{c_name}: {prob*100:.2f}%")
                        
                    # Grad-CAM Explainability Section
                    st.subheader("3. Explainable AI: Grad-CAM Saliency")
                    try:
                        from src.explainability import find_last_conv_layer, make_gradcam_heatmap, generate_gradcam_overlay
                        nested_base, last_conv_name = find_last_conv_layer(model)
                        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_name, pred_index=top_indices[0], base_model=nested_base)
                        heatmap_resized, overlay = generate_gradcam_overlay(original_rgb, heatmap)
                        
                        gcol1, gcol2 = st.columns(2)
                        with gcol1:
                            st.image(heatmap_resized, caption="Grad-CAM Saliency Heatmap", clamp=True, use_container_width=True)
                        with gcol2:
                            st.image(overlay, caption="Superimposed Diagnosis Overlay", use_container_width=True)
                            
                        st.caption("ℹ️ **Interpretation Guide:** Warmer reddish-yellow highlights denote leaf regions that most heavily influenced the neural network's diagnostic decision (e.g. lesion margins, chlorotic halos). Cooler blue regions had negligible impact.")
                    except Exception as e:
                        st.info(f"Grad-CAM explanation currently unavailable for this layer configuration: {e}")
                        
                    # Agronomic Advisory Section
                    st.subheader("4. Agronomic Symptoms & Prevention Guidance")
                    st.markdown(f"**Identified Symptoms:** {info.get('symptoms')}")
                    st.markdown(f"**Preventive Cultural Measures:** {info.get('prevention')}")
                    
                    # Mandatory Ethical / Agronomic Disclaimer
                    st.markdown("""
                    <div class="disclaimer-box">
                        <b>⚠️ Academic & Advisory Disclaimer:</b><br>
                        This artificial intelligence system provides decision-support predictions based on visual leaf characteristics. 
                        It does NOT prescribe chemical pesticides or specific chemical dosages. Farmers and agricultural practitioners must 
                        consult an authorized agronomist, plant pathologist, or local agricultural extension officer (e.g. Krishi Vigyan Kendra) 
                        before initiating chemical interventions.
                    </div>
                    """, unsafe_allow_html=True)

elif nav_choice == "📊 Supported Crops & Diseases":
    st.markdown('<div class="main-header">Supported Crops & Pathology Inventory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Verified distribution across 44 classes from the 53,303 image dataset.</div>', unsafe_allow_html=True)
    
    crops = {}
    for c in class_names:
        parts = c.split("___")
        crop = parts[0]
        dis = parts[1]
        crops.setdefault(crop, []).append(dis)
        
    st.write(f"**Total Documented Crops:** {len(crops)} | **Total Diagnosable Classes:** {len(class_names)}")
    
    st.info("💡 **Maharashtra Regional Context:** Grape, Rice, Tomato, Potato, and Bell Pepper are widely cultivated cash and food crops across Maharashtra. Sugarcane and Onion are not in this benchmark dataset and are not claimed.")
    
    for crop, diseases in crops.items():
        with st.expander(f"🌾 {crop.upper()} ({len(diseases)} condition{'s' if len(diseases)>1 else ''})"):
            for d in diseases:
                full_key = f"{crop}___{d}"
                cond_info = disease_info.get(full_key, {})
                status_color = "🟢" if "healthy" in d.lower() else "🔴"
                st.write(f"- {status_color} **{d}** — *{cond_info.get('symptoms', 'No symptoms listed')}*")

elif nav_choice == "ℹ️ About the Project":
    st.markdown('<div class="main-header">About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    **Project Title:** Smart Crop Disease Detection with Explainable Edge Deployment  
    **Student Name:** DAHIPHALE SAMRADNI ASHOK  
    **Project ID:** A:32  
    **Academic Year:** 2026–2027  
    
    ### Architectural Highlights
    - **Four Compared Neural Architectures:** Custom CNN Baseline, VGG16, ResNet50, MobileNetV2.
    - **Explainable AI (XAI):** Gradient-weighted Class Activation Mapping (Grad-CAM) to build farmer trust.
    - **Edge Deployment Optimization:** Post-training Float32, Float16, and INT8 quantization via TensorFlow Lite.
    - **Dataset Lineage:** 53,303 images comprising PlantVillage, Cassava, and Rice benchmarks.
    """)
