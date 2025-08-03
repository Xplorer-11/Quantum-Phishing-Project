import sqlite3
from datetime import datetime

DATABASE_FILE = "history.db"

def init_db():
    """Initializes the database and creates the history table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                prediction TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

def log_prediction(url: str, prediction: str):
    """Logs a new prediction into the history table."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("INSERT INTO history (url, prediction, timestamp) VALUES (?, ?, ?)", 
                       (url, prediction, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")

def get_recent_history(limit: int = 10):
    """Fetches the most recent prediction records."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        # Use a dictionary cursor to get results as key-value pairs
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT url, prediction, timestamp FROM history ORDER BY timestamp DESC LIMIT ?", (limit,))
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return records
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []

def get_stats():
    """Fetches statistics about phishing sites."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Get the top 5 most frequently reported phishing sites
        cursor.execute("""
            SELECT url, COUNT(url) as count 
            FROM history 
            WHERE prediction='Phishing' 
            GROUP BY url 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_phishing_sites = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return {"top_phishing_sites": top_phishing_sites}
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return {"top_phishing_sites": []}