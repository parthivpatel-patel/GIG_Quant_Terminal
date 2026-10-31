#!/bin/bash
echo "═══════════════════════════════════════════"
echo "  GIGA INVESTMENT GROUP — Starting..."
echo "═══════════════════════════════════════════"

# Start backend
python live_data_server.py &
BACKEND_PID=$!

# Wait for backend
for i in {1..30}; do
    curl -sf http://localhost:5001/api/ping > /dev/null && break
    sleep 2
done

# Start frontend
cd quant_firm/webapp && npx serve -s build -l 3000 2>/dev/null || npm start &

echo "═══════════════════════════════════════════"
echo "  GIG LIVE — Backend :5001 | Frontend :3000"
echo "═══════════════════════════════════════════"

wait $BACKEND_PID
