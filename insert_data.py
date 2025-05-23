import json
import psycopg2
from psycopg2.extras import Json

# Database connection parameters
DB_PARAMS = {
    'dbname': 'your_database',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def insert_data():
    """Read JSON data and insert into PostgreSQL"""
    # Read JSON file
    with open('test_data.json', 'r') as file:
        data = json.load(file)
    
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            for item in data:
                # Convert labels_weight to JSONB format
                labels_weight = Json(item['labels_weight'])
                
                # Insert data into the table
                cur.execute("""
                    INSERT INTO messages (id, message, labels_weight, "from", company)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET message = EXCLUDED.message,
                        labels_weight = EXCLUDED.labels_weight,
                        "from" = EXCLUDED.from,
                        company = EXCLUDED.company
                """, (
                    item['id'],
                    item['message'],
                    labels_weight,
                    item['from'],
                    item['company']
                ))
        
        conn.commit()
        print("Data inserted successfully!")
    
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    insert_data() 