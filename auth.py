import streamlit as st
import hashlib

# temporary user storage (college demo)
users = {}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register():
    st.subheader("📝 Register")
    username = st.text_input("Username", key="reg_user")
    password = st.text_input("Password", type="password", key="reg_pass")

    if st.button("Register"):
        if username == "" or password == "":
            st.warning("Fill all fields")
        elif username in users:
            st.error("User already exists")
        else:
            users[username] = hash_password(password)
            st.success("Registered successfully! Now login.")

def login():
    st.subheader("🔐 Login")
    username = st.text_input("Username", key="log_user")
    password = st.text_input("Password", type="password", key="log_pass")

    if st.button("Login"):
        if username in users and users[username] == hash_password(password):
            st.session_state["authenticated"] = True
            st.success("Login successful")
        else:
            st.error("Invalid username or password")
