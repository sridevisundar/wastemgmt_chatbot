import streamlit as st
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import tensorflow as tf
import numpy as np
from PIL import Image
#from dotenv import load_dotenv

# ✅ Load environment variables
from dotenv import load_dotenv
import os

client = Groq(api_key="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")

# ✅ Ensure compatibility with Windows asyncio
import asyncio
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ Initialize Groq client


# ✅ Load data from JSON file
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON format: Expected a dictionary.")
        return list(data.values())

# ✅ Initialize ChromaDB client and collection
@st.cache_resource
def init_chroma_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="waste_management_rag")

# ✅ Load MobileNetV2 model
@st.cache_resource
def load_mobilenet_model():
    return tf.keras.models.load_model("mobilenetv2_model.h5")

# ✅ Load embeddings model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load data into ChromaDB
def load_into_chroma(data, collection, embedding_model):
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()

    new_data = []
    for idx, entry in enumerate(data):
        doc_id = f"doc_{idx}"
        if doc_id not in existing_ids and isinstance(entry, dict) and "full_text" in entry:
            new_data.append((doc_id, entry))

    if not new_data:
        return

    embeddings = embedding_model.encode(
        [entry["full_text"] for _, entry in new_data],
        batch_size=16,
        show_progress_bar=True
    ).tolist()

    collection.add(
        documents=[entry["full_text"] for _, entry in new_data],
        metadatas=[{
            "title": entry.get("title", ""),
            "summary": entry.get("summary", "")
        } for _, entry in new_data],
        ids=[doc_id for doc_id, _ in new_data],
        embeddings=embeddings
    )

# ✅ Query Groq for text-based answers
def query_groq(prompt):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Use "mixtral" or another Groq model if needed
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API Error: {e}"

# ✅ Get relevant context from ChromaDB
def get_relevant_context(query, collection, embedding_model, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])
    metadata = results.get("metadatas", [[]])

    filtered_docs = [
        doc for i, doc in enumerate(documents[0])
        if any(
            keyword in metadata[0][i].get("title", "").lower() or
            keyword in metadata[0][i].get("summary", "").lower()
            for keyword in ["waste", "recycle", "types", "classification", "categories"]
        )
    ]

    return filtered_docs[:top_k] if filtered_docs else []

# ✅ Answer question using retrieved context
def ask_question(question, collection, embedding_model):
    context = get_relevant_context(question, collection, embedding_model)
    if not context:
        return "⚠ No relevant information found in the database."

    max_context_tokens = 1000
    formatted_context = "\n".join(context)[:max_context_tokens]

    prompt = f"""You are a waste management expert. 
Answer the following question based on the context provided. 
Provide a clear, detailed, and actionable answer. If the waste item requires special handling (e.g., batteries, electronics), suggest appropriate disposal methods:
Context: {formatted_context}
Question: {question}
Answer:"""


    return query_groq(prompt)

# ✅ Classify waste using MobileNetV2
def classify_waste(image, model):
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    img = tf.image.resize(image, (224, 224)) / 255.0
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    disposal_methods = {
        "cardboard": "Cardboard should be disposed of by first removing any non-paper elements like tape, staples, and labels. Flatten the boxes to save space and make them easier to handle. Make sure the cardboard is clean and dry, as wet or greasy cardboard may not be recyclable. Check local recycling guidelines to confirm whether specific types of cardboard, like pizza boxes, are accepted. Finally, place the cardboard in a designated recycling bin or take it to a nearby recycling center.",
        "glass": "Glass should be disposed of by first rinsing it to remove any food or liquid residue. Remove any lids, caps, or corks, as these are often made from different materials and may need to be recycled separately. Check local recycling guidelines to see if certain types of glass, like broken glass or tempered glass, are accepted. If recycling isn’t available, consider reusing glass containers for storage or crafts. Finally, place recyclable glass in a designated glass recycling bin or take it to a nearby recycling center.",
        "metal": "Metal should be disposed of by first cleaning it to remove any food or residue. Separate different types of metal, such as aluminum cans and steel containers, as they may be processed differently. Remove any non-metal parts, like plastic lids or paper labels, if required by local recycling guidelines. Check with your local recycling center to confirm which types of metal are accepted. Finally, place the metal in a designated recycling bin or take it to a scrap metal facility.",
        "paper": "Paper should be disposed of by keeping it clean and dry, as wet or greasy paper cannot be recycled. Remove any staples, plastic windows, or tape from envelopes or packaging. Shred sensitive documents if needed, but avoid shredding paper that doesn’t require it, as shredded paper is harder to recycle. Check local recycling guidelines for specific types of paper, like receipts or coated paper. Finally, place the paper in a designated recycling bin or drop it off at a recycling center.",
        "plastic": "Plastic should be disposed of by first checking the recycling symbol and number to determine if it's recyclable in your area. Rinse out any food or liquid residue and remove any non-plastic components like caps or labels, unless your local guidelines allow them. Avoid recycling plastic bags and film unless your local recycling program accepts them; instead, take them to a drop-off location if available. Try to reduce plastic use by reusing containers or choosing alternatives when possible. Finally, place recyclable plastics in the designated recycling bin or drop them off at a recycling center.",
        "trash": "Trash should be disposed of by first separating recyclable and compostable items to reduce waste. Place non-recyclable and non-compostable items, like certain plastics, broken ceramics, and non-recyclable packaging, into a sturdy trash bag. Seal the bag securely to prevent leaks or spills. Hazardous materials, like batteries or chemicals, should not be thrown in the regular trash and need to be handled according to local guidelines. Finally, place the trash bag in a designated waste bin for collection or disposal."
    }

    suggestion = disposal_methods.get(predicted_class, "Unknown waste type.")
    return predicted_class, confidence, suggestion

# ✅ Streamlit UI
def main():
    st.title("♻️ Waste Management Chatbot")
    st.write("Ask any question related to waste management or upload an image to classify waste.")

    # Load data and initialize models
    file_path = "wikipedia_waste_management_data.json"
    data = load_data(file_path)
    collection = init_chroma_db()
    embedding_model = get_embedding_model()
    mobilenet_model = load_mobilenet_model()

    if collection.count() == 0:
        with st.spinner("🗄 Loading data into ChromaDB..."):
            load_into_chroma(data, collection, embedding_model)
            st.success(f"✅ Loaded {len(data)} articles into ChromaDB on startup!")

    # ✅ Tabs for text or image input
    tab1, tab2 = st.tabs(["💬 Ask a Question", "📸 Upload Image"])

    # ✅ Text-based Q&A
    with tab1:
        question = st.text_input(
            "💬 Ask a question:",
            key="question_input",
            placeholder="Type your question here...",
            max_chars=300
        )
        if st.button("Ask"):
            with st.spinner("🤖 Thinking..."):
                response = ask_question(question, collection, embedding_model)
                st.write(f"**🤖 Answer:** {response}")

    # ✅ Image-based waste classification
    with tab2:
        uploaded_file = st.file_uploader("📸 Upload an image of waste:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Convert PIL image to TensorFlow tensor
            image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)

            if st.button("Classify"):
                with st.spinner("🧪 Classifying..."):
                    predicted_class, confidence, suggestion = classify_waste(image_tensor, mobilenet_model)
                    st.write(f"**🔍 Detected:** {predicted_class} ({confidence:.2%} confidence)")
                    st.write(f"**💡 Suggested Disposal:** {suggestion}")

if __name__ == "__main__":
    main()

