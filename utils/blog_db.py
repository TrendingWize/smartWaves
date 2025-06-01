# utils/blog_db.py
import sqlite3


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
