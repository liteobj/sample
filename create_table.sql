CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    message TEXT,
    labels_weight JSONB,  -- Will store as {"label_name": weight}
    "from" VARCHAR(255),
    company VARCHAR(255)
); 