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
ORDER BY total_weight DESC; 