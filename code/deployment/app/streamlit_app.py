import streamlit as st
import requests

st.title("Technical Assistant")

user_question = st.text_area("Enter your question")

if st.button("Analyze"):
    if user_question.strip() == "":
        st.warning("Enter your question")
    else:
        response_class = requests.post(
            "http://api:8000/classify", json={"text": user_question}
        )
        if response_class.status_code == 200:
            classifications = response_class.json()["classification"]
            st.subheader("Question tag:")
            for cls in classifications:
                st.write(f"{cls['label']} (score: {cls['score']:.2f})")
        else:
            st.error("Smth went wrong")


        response_gen = requests.post(
            "http://api:8000/generate", json={"text": user_question}
        )
        if response_gen.status_code == 200:
            answer = response_gen.json()["answer"]
            st.subheader("Possible answer")
            st.write(answer)
        else:
            st.error("Smth went wrong")
