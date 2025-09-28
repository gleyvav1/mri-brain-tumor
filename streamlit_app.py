# streamlit_app.py — Demo MRI Tumor (CPU)
# Sube una imagen, predice YES/NO y muestra Grad-CAM + caja (solo si YES)

import os, io, json, math
from pathlib import Path

# --- Evitar problemas de permisos y telemetría en plataformas gestionadas ---
os.environ["HOME"] = str(Path.cwd())
(Path.cwd()/".streamlit").mkdir(parents=True, exist_ok=True)
os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import streamlit as st

# =======================
# Configuración y helpers
# =======================
MODEL_PATH = Path("resnet18_best.pt")                  # pesos del modelo
THR_PATH   = Path("best_threshold.json")               # umbral opcional
IMG_SIZE   = 224

@st.cache_resource(show_spinner=False)
def load_model_and_misc():
    # Modelo (ResNet18 con 2 clases)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    assert MODEL_PATH.exists(), f"No encuentro {MODEL_PATH}"
    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.eval()

    # Umbral
    thr = 0.5
    if THR_PATH.exists():
        try:
            info = json.load(open(THR_PATH))
            thr = float(info["thresholds"]["recall_priority"]["threshold"])
        except Exception:
            pass

    # Transforms (igual que en entrenamiento)
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    return m, tfm, thr

def pil_to_rgb_np(pil_img):
    return np.array(pil_img.convert("RGB"))

def tensor_to_img_uint8(t):
    mean = np.array([0.485,0.456,0.406]).reshape(3,1,1)
    std  = np.array([0.229,0.224,0.225]).reshape(3,1,1)
    x = t.detach().cpu().numpy()
    x = (x * std + mean).clip(0,1)
    x = (x*255).astype(np.uint8).transpose(1,2,0)
    return x

def gradcam_yes_overlay(model, x_tensor_1x3xHxW, orig_rgb):
    """
    Calcula Grad-CAM para la clase 'yes' (índice 1) y devuelve overlay en RGB
    + dibuja una caja (contorno mayor) si la hay.
    """
    feats = {}
    target_layer = model.layer4[-1].conv2

    def fwd_hook(m, inp, out): feats['a'] = out.detach()
    def bwd_hook(m, gin, gout): feats['g'] = gout[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    with torch.set_grad_enabled(True):
        x_req = x_tensor_1x3xHxW.clone().requires_grad_(True)
        logits = model(x_req)
        logits[0, 1].backward(retain_graph=True)

    h1.remove(); h2.remove()

    assert 'a' in feats and 'g' in feats, "No se capturaron activaciones/gradientes."
    A = feats['a'][0]                      # CxHxW
    G = feats['g'][0]                      # CxHxW
    w = G.mean(dim=(1,2), keepdim=True)    # Cx1x1
    cam = (w * A).sum(0).clamp(min=0).cpu().numpy()
    cam = cam / (cam.max() + 1e-8)

    h0, w0 = orig_rgb.shape[0], orig_rgb.shape[1]
    heat = cv2.resize(cam, (w0, h0))
    hm_u8 = (heat*255).astype(np.uint8)

    overlay = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = (0.35*overlay + 0.65*orig_rgb[..., ::-1]).astype(np.uint8)  # mezcla BGR con RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Caja por Otsu (si hay contornos)
    _, th = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x1, y1, wc, hc = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cv2.rectangle(overlay, (x1,y1), (x1+wc, y1+hc), (255,255,255), 2)

    return overlay

# =========
# Interfaz
# =========
st.set_page_config(page_title="MRI Brain Tumor — Demo", page_icon="🧠", layout="centered")
st.title("MRI Brain Tumor — Clasificación y Grad-CAM")
st.caption("Consulta de ejemplo (no para uso clínico).")

model, tfm, threshold = load_model_and_misc()

with st.sidebar:
    st.header("Parámetros")
    st.write(f"Umbral de decisión (YES si prob ≥ umbral): **{threshold:.2f}**")

uploaded = st.file_uploader("Sube una imagen (PNG/JPG/TIFF)", type=["png","jpg","jpeg","tif","tiff"])
btn = st.button("Analizar", type="primary", disabled=(uploaded is None))

if btn and uploaded is not None:
    try:
        pil = Image.open(uploaded).convert("L")
    except Exception as e:
        st.error(f"No pude abrir la imagen: {e}")
        st.stop()

    # Preprocesar
    x = tfm(pil).unsqueeze(0)  # 1x3x224x224
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1).numpy()[0]
    p_no, p_yes = float(probs[0]), float(probs[1])
    pred = "yes" if p_yes >= threshold else "no"

    # Título estilo pedido
    st.subheader(f"Pred: **{pred.upper()}** | prob_yes={p_yes:.3f} | thr={threshold:.2f}")

    orig_rgb = pil_to_rgb_np(pil)

    if pred == "no":
        # Mostrar solo la imagen original (sin Grad-CAM)
        st.image(orig_rgb, use_column_width=True)
        # Botón de descarga
        out_name = Path(uploaded.name).stem + f"_pred-no_py{p_yes:.3f}.png"
        buf = io.BytesIO()
        Image.fromarray(orig_rgb).save(buf, format="PNG"); buf.seek(0)
        st.download_button("Descargar PNG", buf, file_name=out_name, type="secondary")
    else:
        # Grad-CAM + caja
        overlay = gradcam_yes_overlay(model, x, orig_rgb)
        st.image(overlay, use_column_width=True)
        # Botón de descarga
        out_name = Path(uploaded.name).stem + f"_pred-yes_py{p_yes:.3f}.png"
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG"); buf.seek(0)
        st.download_button("Descargar PNG", buf, file_name=out_name, type="secondary")

# Footer
st.markdown("---")
st.caption("© Tu proyecto — Clínica La Luz. Educación y demostración.")
