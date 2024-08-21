import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import base64  # Import the base64 module

# Function to load a local image and convert it to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

book_icon_base64 = get_base64_image("book_icon.png")

# Load the precomputed document vectors and metadata
vector_df = pd.read_csv('document_vectors.csv')
asbab_nuzul = pd.read_excel('asbabun_nuzul.xlsx')

# Add a unique 'No' column if it doesn't exist
if 'No' not in vector_df.columns:
    vector_df.insert(0, 'No', range(1, 1 + len(vector_df)))

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert the vectors from DataFrame to a NumPy array
document_vectors = vector_df.drop(columns=['No', 'topic', 'document', 'surah']).values
docs = vector_df['document'].tolist()
topics = vector_df['topic'].tolist()

# Function to perform k-NN search
def knn_search(input_keyword, model, document_vectors, vector_df, k=10):
    # Encode the input keyword
    vector_of_input_keyword = model.encode(input_keyword)
    
    # Compute cosine similarity
    similarities = cosine_similarity([vector_of_input_keyword], document_vectors)[0]
    
    # Get the indices of the top k most similar documents
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    # Retrieve the top k topics and documents
    top_k_topics = [topics[idx] for idx in top_k_indices]
    top_k_docs = [vector_df.iloc[idx]['No'] for idx in top_k_indices]
    
    # Count the occurrences of each topic
    topic_counts = pd.Series(top_k_topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    # Calculate the recall percent
    topic_counts['Recall Percent (%)'] = (topic_counts['Count'] / 539) * 100
    
    # Add the Document numbers related to each topic
    topic_counts['Document No'] = topic_counts['Topic'].apply(
        lambda topic: ", ".join(map(str, [top_k_docs[i] for i, t in enumerate(top_k_topics) if t == topic]))
    )
    
    return topic_counts

# Hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load user data from CSV
def load_users():
    if not os.path.exists("users.csv"):
        return pd.DataFrame(columns=["username", "password"])
    return pd.read_csv("users.csv")

# Save user data to CSV
def save_user(username, password):
    users = load_users()
    new_user = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)

# Authenticate user
def authenticate(username, password):
    users = load_users()
    if username in users["username"].values:
        stored_password = users[users["username"] == username]["password"].values[0]
        return stored_password == hash_password(password)
    return False

# Streamlit multipage app setup with login and registration
def main():

    # Custom CSS
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #8B4513; /* Brown background for the sidebar */
            color: #FFFFFF; /* White text color for the sidebar */
        }
        .css-1aumxhk, .css-1v3fvcr {
            background-color: #FFFFFF; /* White background for main area */
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Times New Roman', Times, serif;
        }
        </style>
        """, unsafe_allow_html=True)
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in: 
        st.sidebar.markdown(f"""
            <h1 style="font-family:Times New Roman; color:white;">
                <img src="data:image/png;base64,{book_icon_base64}" style="vertical-align: middle; width: 3em; height: 3em;"/>
                BOOKTI
            </h1>
        """, unsafe_allow_html=True)
        st.sidebar.markdown(f'<h2 style="font-family:Arial; color:white;">Selamat Datang! {st.session_state.user}</h2>', unsafe_allow_html=True)
        page = st.sidebar.selectbox("Pilih Halaman", ["TOPIK CARIAN", "PUSTAKA"])
        
        if page == "TOPIK CARIAN":
            search_page()
        elif page == "PUSTAKA":
            view_all_data_page()
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = ""
            st.experimental_rerun()
    else:
        if "page" not in st.session_state or st.session_state.page == "Login":
            login_page()
        elif st.session_state.page == "Register":
            register_page()

def login_page():
    st.markdown(f"""
        <h1 style="font-family:Times New Roman;">
            <img src="data:image/png;base64,{book_icon_base64}" style="vertical-align: middle; width: 4em; height: 4em;"/>
            BOOKTI
        </h1>
    """, unsafe_allow_html=True)
    st.write("Pustaka Asbab Al-Nuzul")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.user = username
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
    st.write("Don't have BOOKTI account?")
    if st.button("Create BOOKTI Account"):
        st.session_state.page = "Register"
        st.experimental_rerun()

def register_page():
    st.title("Create Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if new_password != confirm_password:
            st.error("Passwords do not match")
        else:
            users = load_users()
            if new_username in users["username"].values:
                st.error("Username already exists")
            else:
                save_user(new_username, new_password)
                st.success("Account created successfully! Please login.")
                st.session_state.page = "Login"
                st.experimental_rerun()

    if st.button("Back"):
        st.session_state.page = "Login"
        st.experimental_rerun()

def search_page():
    st.title("Cari Topik Asbab Al-Nuzul ")
    
    # First search bar for topic search
    search_query = st.text_input("Enter your search query")
    if st.button("Search"):
        if search_query:
            st.session_state.search_query = search_query
            st.session_state.search_results = knn_search(search_query, model, document_vectors, vector_df, k=10)
    
    # Display search results for topic search
    if "search_results" in st.session_state:
        st.subheader(f"Search Results for '{st.session_state.search_query}'")
        st.dataframe(st.session_state.search_results)
    
    # Second search bar for document number search
    search_no = st.text_input("Enter document 'No' to search")
    if st.button("Search Document No"):
        if search_no:
            try:
                filtered_df = asbab_nuzul[asbab_nuzul['No'] == int(search_no)]
                if not filtered_df.empty:
                    st.write("Search Results:")
                    st.dataframe(filtered_df)
                else:
                    st.write("No document found with that 'No'")
            except ValueError:
                st.write("Please enter a valid document 'No'")
        else:
            st.write("Please enter a document 'No'")

def view_all_data_page():
    st.title("PUSTAKA")
    st.write("Surah/ Terjemahan/ Asbab Al-Nuzul")
    st.dataframe(asbab_nuzul)

if __name__ == "__main__":
    main()
