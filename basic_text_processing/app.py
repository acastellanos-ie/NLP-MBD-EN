import streamlit as st
import tiktoken
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="NLP 2026: The Tokenizer Playground", layout="wide")

st.title("✂️ The Tokenizer Playground")
st.markdown("""
Explore how LLM's really 'see' the text. Check how common words use a unique token but more complex words are broken into pieces.
""")

# Selector de modelo
col1, col2 = st.columns([1, 3])
with col1:
    model_name = st.selectbox(
        "Pick the tokenizer (Model):",
        ["gpt-4o", "gpt-3.5-turbo", "text-davinci-003"],
        index=0
    )
    
    # Coste aproximado por 1M tokens (ejemplo precios 2025/26)
    pricing = {
        "gpt-4o": 5.00,
        "gpt-3.5-turbo": 0.50,
        "text-davinci-003": 20.00
    }
    
    st.info(f"Estimated Cost: ${pricing[model_name]} / 1M tokens")

# Área de texto
text_input = st.text_area(
    "Input your text here:",
    value="NLP 2026 course is amazing. 🚀 Strawberry.",
    height=150
)

if text_input:
    # Lógica de Tokenización
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text_input)
    num_tokens = len(tokens)
    
    # Métricas
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", len(text_input))
    c1.metric("Words (aprox)", len(text_input.split()))
    c2.metric("Tokens", num_tokens)
    
    ratio = num_tokens / len(text_input.split()) if len(text_input.split()) > 0 else 0
    c3.metric("Ratio Tokens/Word", f"{ratio:.2f}x")

    st.divider()

    # Visualización de Tokens con Colores
    st.subheader("Token Visualization")
    
    # Paleta de colores para alternar
    colors = ["#FFD700", "#ADFF2F", "#00BFFF", "#FF69B4", "#FFA500"]
    
    html_content = ""
    for i, token_id in enumerate(tokens):
        # Decodificar el token individualmente para mostrarlo
        word = encoding.decode([token_id])
        # Reemplazar saltos de línea para que se vean en HTML
        word_display = word.replace("\n", "↵").replace(" ", "&nbsp;")
        
        color = colors[i % len(colors)]
        tooltip = f"Token ID: {token_id}"
        
        html_content += f"""
        <span style="background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 4px; display: inline-block; font-family: monospace; font-size: 1.2em;" title="{tooltip}">
            {word_display}
        </span>
        """
    
    st.markdown(html_content, unsafe_allow_html=True)
    
    st.divider()
    
    # Tabla de detalles
    with st.expander("See detailed ID table"):
        token_data = {
            "Token (Text)": [encoding.decode([t]) for t in tokens],
            "Token ID (Number)": tokens
        }
        st.dataframe(token_data)

else:
    st.warning("Escribe algo arriba para comenzar.")
