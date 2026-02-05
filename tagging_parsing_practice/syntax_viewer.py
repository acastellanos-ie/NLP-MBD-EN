import streamlit as st
import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import pandas as pd
import numpy as np

# Configuración
st.set_page_config(page_title="Session 3: Syntax & Attention", layout="wide")
st.title("🧠 Linguistic Structure: Trees vs. Attention")

st.markdown("""
En esta sesión comparamos cómo veíamos el lenguaje antes (**Árboles sintácticos explícitos**) 
vs. cómo lo ven los modelos modernos (**Mecanismos de Atención**).
""")

# Cargar recursos (Cacheado para velocidad)
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_model():
    # Usamos un modelo BERT pequeño para visualizar la atención real
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

nlp = load_spacy()
tokenizer, model = load_model()

# Input
text = st.text_input("Escribe una frase compleja:", "The quick brown fox jumps over the lazy dog.")

if text:
    # --- PESTAÑA 1: VISIÓN CLÁSICA (SPACY) ---
    st.header("1. La Visión Clásica: Dependency Parsing")
    st.info("Esto es lo que lingüistas y modelos pre-2018 (como Spacy) calculan explícitamente.")
    
    doc = nlp(text)
    
    # Visualizador de Spacy
    # Usamos HTML wrapper para centrarlo
    html = displacy.render(doc, style="dep", options={"compact": True, "distance": 100})
    st.write(html, unsafe_allow_html=True)
    
    # Tabla de dependencias
    data = []
    for token in doc:
        data.append([token.text, token.pos_, token.dep_, token.head.text])
    
    st.dataframe(pd.DataFrame(data, columns=["Token", "POS Tag", "Dependency", "Head (Padre)"]), use_container_width=True)

    st.divider()

    # --- PESTAÑA 2: VISIÓN MODERNA (ATTENTION) ---
    st.header("2. La Visión Moderna: Attention Mechanism (BERT)")
    st.info("Los Transformers no construyen árboles. Calculan 'Atención': ¿Cuánto 'mira' una palabra a las otras?")
    
    # Procesar con BERT
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    attention = outputs.attentions # Tupla de (Capas) x (Batch, Heads, Seq, Seq)
    
    # Selector de Capa y Cabeza
    c1, c2 = st.columns(2)
    layer = c1.slider("Selecciona la Capa (Layer) del modelo:", 0, 11, 10)
    head = c2.slider("Selecciona la Cabeza (Head) de atención:", 0, 11, 5)
    
    # Obtener matriz de atención para la capa/cabeza seleccionada
    # Shape: [Seq_Len, Seq_Len]
    att_matrix = attention[layer][0, head].detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Visualización Heatmap simple
    # Creamos un DataFrame para el heatmap
    df_att = pd.DataFrame(att_matrix, index=tokens, columns=tokens)
    
    st.write(f"**Mapa de Atención (Capa {layer}, Cabeza {head})**")
    st.markdown("Busca patrones: ¿Ves alguna cabeza que conecte adjetivos con sustantivos? ¿O determinantes con nombres?")
    
    # Usamos st.dataframe con gradiente de color para simular heatmap interactivo
    st.dataframe(
        df_att.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}"),
        height=400,
        use_container_width=True
    )
    
    st.caption("Nota: Los tokens [CLS] y [SEP] son especiales de BERT. Ignóralos por ahora.")

else:
    st.write("Esperando input...")
