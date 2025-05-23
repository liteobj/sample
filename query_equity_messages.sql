SELECT 
    id,
    message,
    "from",
    company,
    (labels_weight->>'Equity')::integer as equity_weight
FROM messages
WHERE labels_weight ? 'Equity'
ORDER BY equity_weight DESC; 