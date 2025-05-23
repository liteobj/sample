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

def get_equity_trade_messages():
    """Get all messages with both Equity and Trade labels sorted by combined weight"""
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
                    (labels_weight->>'Equity')::integer as equity_weight,
                    (labels_weight->>'Trade')::integer as trade_weight,
                    ((labels_weight->>'Equity')::integer + (labels_weight->>'Trade')::integer) as total_weight
                FROM messages
                WHERE labels_weight ? 'Equity' 
                AND labels_weight ? 'Trade'
                ORDER BY total_weight DESC
            """)
            
            # Fetch all results
            results = cur.fetchall()
            
            if not results:
                print("No messages found with both Equity and Trade labels")
                return
            
            # Print results in a nice table format
            headers = ['ID', 'Message', 'From', 'Company', 'Equity Weight', 'Trade Weight', 'Total Weight']
            print("\nMessages with both Equity and Trade labels (sorted by total weight):")
            print(tabulate(results, headers=headers, tablefmt='grid'))
            
    except Exception as e:
        print(f"Error querying data: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    get_equity_trade_messages() 