import psycopg2
from tabulate import tabulate

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

def get_equity_messages():
    """Get all messages with Equity label sorted by weight"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    message,
                    "from",
                    company,
                    (labels_weight->>'Equity')::integer as equity_weight
                FROM messages
                WHERE labels_weight ? 'Equity'
                ORDER BY equity_weight DESC
            """)
            
            # Fetch all results
            results = cur.fetchall()
            
            if not results:
                print("No messages found with Equity label")
                return
            
            # Print results in a nice table format
            headers = ['ID', 'Message', 'From', 'Company', 'Equity Weight']
            print("\nMessages with Equity label (sorted by weight):")
            print(tabulate(results, headers=headers, tablefmt='grid'))
            
    except Exception as e:
        print(f"Error querying data: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    get_equity_messages() 