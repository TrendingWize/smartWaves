# Utility functions (e.g., auth, DB, hashing)
# smart_waves/utils.py
import streamlit as st
import sqlite3
import hashlib
import uuid # For session tokens

DATABASE_NAME = 'users.db'

# --- Database Setup ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_user_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_approved INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0,
            current_session_token TEXT
        )
    ''')
    # Ensure admin user exists
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    admin = cursor.fetchone()
    if not admin:
        hashed_password = hash_password("1234")
        cursor.execute(
            "INSERT INTO users (username, password_hash, is_approved, is_admin) VALUES (?, ?, ?, ?)",
            ("admin", hashed_password, 1, 1)
        )
    conn.commit()
    conn.close()

# --- Password Hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password_hash, provided_password):
    return stored_password_hash == hash_password(provided_password)

# --- User Management ---
def add_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hashed_password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError: # Username already exists
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_unapproved_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE is_approved = 0 AND is_admin = 0")
    users = cursor.fetchall()
    conn.close()
    return users

def approve_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_approved = 1 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def update_session_token(username, token):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET current_session_token = ? WHERE username = ?", (token, username))
    conn.commit()
    conn.close()

# --- Session State Initialization ---
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False
    if 'session_token' not in st.session_state: # For this browser session
        st.session_state.session_token = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home" # Default page

# --- Login / Logout ---
def login_user(username, password):
    user = get_user(username)
    if user and verify_password(user['password_hash'], password):
        if not user['is_approved']:
            st.error("Your account is pending approval from an administrator.")
            return False
        
        # Generate a new session token for this login
        new_session_token = str(uuid.uuid4())
        update_session_token(username, new_session_token)

        st.session_state.logged_in = True
        st.session_state.username = user['username']
        st.session_state.is_admin = bool(user['is_admin'])
        st.session_state.session_token = new_session_token # Store in browser session
        st.success(f"Welcome back, {username}!")
        st.rerun() # Rerun to update UI
        return True
    st.error("Invalid username or password.")
    return False

def logout_user():
    if st.session_state.username:
        update_session_token(st.session_state.username, None) # Clear token in DB
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.is_admin = False
    st.session_state.session_token = None
    st.success("You have been logged out.")
    st.rerun()

# --- Concurrent Login Check ---
def check_concurrent_login():
    if st.session_state.logged_in and st.session_state.username and st.session_state.session_token:
        user = get_user(st.session_state.username)
        if user and user['current_session_token'] != st.session_state.session_token:
            st.warning("You have been logged out because this account was logged in from another device/browser.")
            logout_user() # This will call rerun
            return True # Was logged out
    return False # Not logged out by this check
