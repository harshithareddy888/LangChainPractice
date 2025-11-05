import langchain_helper as lch
import streamlit as st

st.title("Pet Name Generator")  
user_animal_type=st.sidebar.selectbox("What type of animal do you have?", ["dog", "cat", "hamster", "parrot", "rabbit"])
user_pet_color=st.sidebar.text_area("What color is your pet?",max_chars=15)


if user_pet_color:
    response= lch.generate_pet_name(user_animal_type, user_pet_color)
    st.text(response['pet_name'])