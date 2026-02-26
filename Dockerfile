# ---------------- Base Image ----------------
FROM python:3.10-slim

# ---------------- Environment ----------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Replace with your actual Resend API key on the server
ENV RESEND_API_KEY=dummy_replace_me

WORKDIR /app

# ---------------- System Dependencies ----------------
# Required for:
# - OpenCV
# - PaddleOCR
# - PIL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------- Python Dependencies ----------------
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------- App Code ----------------
COPY . .

# ---------------- Runtime ----------------
EXPOSE 8000

# Railway runs containers, not serverless
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]