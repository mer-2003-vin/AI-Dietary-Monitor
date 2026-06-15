import sqlite3
import os
import hashlib

DB_PATH = "meals.db"

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def init_db():
    """Initializes the SQLite database and creates the tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    """)
    
    # Create meals table with username column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT NOT NULL,
            meal_type TEXT NOT NULL,
            food TEXT NOT NULL,
            calories REAL DEFAULT 0.0,
            protein_g REAL DEFAULT 0.0,
            carbs_g REAL DEFAULT 0.0,
            fat_g REAL DEFAULT 0.0,
            fiber_g REAL DEFAULT 0.0,
            sugar_g REAL DEFAULT 0.0,
            sodium_mg REAL DEFAULT 0.0,
            iron_mg REAL DEFAULT 0.0,
            calcium_mg REAL DEFAULT 0.0,
            vitamin_c_mg REAL DEFAULT 0.0
        )
    """)
    
    # For backwards compatibility, check if the username column needs to be added
    try:
        cursor.execute("ALTER TABLE meals ADD COLUMN username TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass
        
    conn.commit()
    conn.close()

def register_user(username, password):
    """Registers a new user. Returns True if successful, False if username already exists."""
    username = username.strip().lower()
    if not username or not password:
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False
        
    # Insert new user
    pass_hash = hash_password(password)
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pass_hash))
        conn.commit()
        success = True
    except Exception:
        success = False
        
    conn.close()
    return success

def authenticate_user(username, password):
    """Authenticates user credentials. Returns True if successful, otherwise False."""
    username = username.strip().lower()
    if not username or not password:
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return False
        
    stored_hash = row[0]
    return stored_hash == hash_password(password)

def add_meal_to_db(username, date_str, meal_type, food_name, nutrition):
    """Inserts a new meal record into the database associated with a username."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract nutrition values, defaulting to 0.0 if not present
    calories = float(nutrition.get("calories", 0.0))
    protein_g = float(nutrition.get("protein_g", 0.0))
    carbs_g = float(nutrition.get("carbs_g", 0.0))
    fat_g = float(nutrition.get("fat_g", 0.0))
    fiber_g = float(nutrition.get("fiber_g", 0.0))
    sugar_g = float(nutrition.get("sugar_g", 0.0))
    sodium_mg = float(nutrition.get("sodium_mg", 0.0))
    iron_mg = float(nutrition.get("iron_mg", 0.0))
    calcium_mg = float(nutrition.get("calcium_mg", 0.0))
    vitamin_c_mg = float(nutrition.get("vitamin_c_mg", 0.0))
    
    cursor.execute("""
        INSERT INTO meals (
            username, date, meal_type, food, calories, protein_g, carbs_g, fat_g, 
            fiber_g, sugar_g, sodium_mg, iron_mg, calcium_mg, vitamin_c_mg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username.strip().lower(), date_str, meal_type, food_name, calories, protein_g, carbs_g, fat_g,
        fiber_g, sugar_g, sodium_mg, iron_mg, calcium_mg, vitamin_c_mg
    ))
    conn.commit()
    conn.close()

def get_all_meals_from_db(username):
    """Retrieves all meal logs from the database for a specific user, returned as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, meal_type, food, calories, protein_g, carbs_g, fat_g,
               fiber_g, sugar_g, sodium_mg, iron_mg, calcium_mg, vitamin_c_mg
        FROM meals
        WHERE username = ?
        ORDER BY id ASC
    """, (username.strip().lower(),))
    rows = cursor.fetchall()
    conn.close()
    
    meals = []
    columns = [
        "date", "meal_type", "food", "calories", "protein_g", "carbs_g", "fat_g",
        "fiber_g", "sugar_g", "sodium_mg", "iron_mg", "calcium_mg", "vitamin_c_mg"
    ]
    for row in rows:
        meals.append(dict(zip(columns, row)))
    return meals

def clear_all_meals_from_db(username):
    """Clears all logs from the database for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM meals WHERE username = ?", (username.strip().lower(),))
    conn.commit()
    conn.close()
