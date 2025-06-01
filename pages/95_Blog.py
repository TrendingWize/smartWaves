# --- blog_db.py ---
import sqlite3
from utils.blog_db import get_all_blog_posts
from datetime import datetime

from utils.blog_db import get_all_blog_posts, init_blog_db

init_blog_db()


def init_blog_db():
    conn = sqlite3.connect("blog_posts.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS blog_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            author TEXT,
            date TEXT,
            tags TEXT,
            content TEXT,
            image_url TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_blog_post(title, author, date, tags, content, image_url):
    conn = sqlite3.connect("blog_posts.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO blog_posts (title, author, date, tags, content, image_url)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (title, author, date, tags, content, image_url))
    conn.commit()
    conn.close()

def get_all_blog_posts():
    conn = sqlite3.connect("blog_posts.db")
    c = conn.cursor()
    c.execute("SELECT * FROM blog_posts ORDER BY date DESC")
    posts = c.fetchall()
    conn.close()
    return posts


# --- pages/07_Blog.py ---
import streamlit as st
from utils.blog_db import get_all_blog_posts
from datetime import datetime

st.set_page_config(page_title="Smart Waves Blog", layout="wide")

st.markdown("""
    <style>
    .blog-post {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: 0.3s ease-in-out;
    }
    .blog-post:hover {
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .blog-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #007bff;
    }
    .blog-meta {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📝 Smart Waves Blog")
st.caption("Insights, news, and updates from the Smart Waves platform.")

posts = get_all_blog_posts()

for post in posts:
    id, title, author, date, tags, content, image_url = post
    with st.container():
        st.markdown("""
        <div class='blog-post'>
        <div class='blog-title'>{}</div>
        <div class='blog-meta'>By {} • {} • {}</div>
        </div>
        """.format(
            title, author, datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y"), tags
        ), unsafe_allow_html=True)

        if image_url:
            st.image(image_url, use_column_width=True)

        with st.expander("Read more"):
            st.markdown(content, unsafe_allow_html=True)

        st.markdown("---")


# --- 99_Admin_Panel.py (insert section for blog post creation) ---
import streamlit as st
from datetime import date

init_blog_db()

st.subheader("📝 Create a Blog Post")
with st.form("blog_form"):
    title = st.text_input("Title")
    author = st.text_input("Author", value="Admin")
    date_val = st.date_input("Date", value=date.today())
    tags = st.text_input("Tags (comma-separated)")
    content = st.text_area("Content", height=300)
    image_url = st.text_input("Optional Image URL")
    submitted = st.form_submit_button("Post Blog")

    if submitted:
        insert_blog_post(title, author, date_val.strftime("%Y-%m-%d"), tags, content, image_url)
        st.success("Blog post published!")
