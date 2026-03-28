import os
import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from matplotlib.colors import TABLEAU_COLORS 
from pathlib import Path

# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.4   # 🔥 Fixed confidence (no slider)
parent_root = Path(__file__).parent.absolute().__str__()
model_onnx_path = os.path.join(parent_root, "yolov7-p6-bonefracture.onnx")
device = "cpu"   # safer than cuda

h, w = 640, 640

# ---------------- UI SETTINGS ----------------
st.set_page_config(page_title="Bone Fracture Detection", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🦴 Bone Fracture Detection</h1>
    <p style='text-align: center;'>Upload an X-ray image to detect fractures</p>
""", unsafe_allow_html=True)

# ---------------- COLORS ----------------
def color_list():
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]

colors = color_list()

# ---------------- HELPERS ----------------
def load_img(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)[..., ::-1]

def preproc(img):
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32).transpose(2, 0, 1)/255
    return np.expand_dims(img, axis=0)

def model_inference(model_path, image_np):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: image_np})[0][:, :6]

def xyxy2xywhn(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def xywhn2xyxy(bbox, H, W):
    x, y, w, h = bbox
    return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

def post_process(img, output):
    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]

    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7:"softtissue", 8:"text"
    }

    img = img.astype(np.uint8)
    H, W = img.shape[:2]
    label_txt = ""

    for i in range(len(det_bboxes)):
        if det_scores[i] > CONF_THRESHOLD:
            bbox = det_bboxes[i]
            label = int(det_labels[i])

            bbox = xyxy2xywhn(bbox, h, w)
            label_txt += f"{label} {det_scores[i]:.2f}\n"

            bbox = xywhn2xyxy(bbox, H, W)
            x1, y1, x2, y2 = map(int, bbox)

            color = colors[label]
            text = f"{id2names[label]} {det_scores[i]:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return img, label_txt

# ---------------- MAIN ----------------
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    img = load_img(uploaded_file)

    ##st.image(img, caption="Original Image", use_column_width=True)

    with st.spinner("🔍 Detecting fractures..."):
        img_pp = preproc(img)
        out = model_inference(model_onnx_path, img_pp)
        result_img, result_txt = post_process(img, out)

    st.success("✅ Detection Complete")

    st.image(result_img, caption="Detected Output", width = 500)

    col1, col2 = st.columns(2)

    col1.download_button(
        "📥 Download Image",
        data=cv2.imencode(".png", result_img[..., ::-1])[1].tobytes(),
        file_name="prediction.png"
    )

    col2.download_button(
        "📥 Download Labels",
        data=result_txt,
        file_name="prediction.txt"
    )