FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose the port Streamlit uses
EXPOSE 8080

# Start command — PORT env var is set by Railway/Render/HF Spaces
CMD streamlit run app.py \
    --server.port ${PORT:-8080} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false