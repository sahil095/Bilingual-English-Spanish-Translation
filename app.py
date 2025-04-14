import streamlit as st
import torch
from translation_model import Transformer, Vocab, tokenize, create_subsequent_mask
from decode_utils import translate

st.title("🌍 English to Spanish Translator (Transformer)")

# Load vocab & model
src_vocab = torch.load("eng_vocab.pth", map_location="cpu")
tgt_vocab = torch.load("spa_vocab.pth", map_location="cpu")
model = Transformer(len(src_vocab), len(tgt_vocab))
model.load_state_dict(torch.load("transformer_model.pth", map_location="cpu"))
model.eval()

def main():
    input_text = st.text_input("Enter English text:", "I love programming")
    decoding_method = st.selectbox("Select decoding method", ["beam", "greedy"])
    beam_width = st.slider("Beam width (for beam search only)", min_value=1, max_value=10, value=5)

    if st.button("Translate"):
        translation = translate(model, input_text, src_vocab, tgt_vocab, method=decoding_method, beam_width=beam_width)
        st.markdown(f"### 📝 Spanish Translation:\n**{translation.title()}**")

if __name__ == "__main__":
    main()
