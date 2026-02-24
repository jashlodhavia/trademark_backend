# trademark_backend

## Deploying to an AWS Ubuntu Instance

These steps assume you have an AWS EC2 instance running Ubuntu 22.04 (or a similar recent version) and that you have SSH access with a key pair.

### 1. Connect to the instance
```bash
ssh -i /path/to/your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 2. Install required dependencies
```bash
# Update package lists
sudo apt update

# Install Docker
sudo apt install -y docker.io
sudo systemctl enable --now docker

# Install Git (if not already installed)
sudo apt install -y git
```

### 3. Clone the repository
```bash
# Choose a directory for the project
mkdir -p ~/trademark_backend && cd ~/trademark_backend

# Clone the repo (replace with your repo URL if different)
git clone https://github.com/jashlodhavia/trademark_backend.git .
```

### 4. (Optional) Set up a Python virtual environment
If you prefer to run the FastAPI app directly without Docker:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Build and run the Docker container
```bash
# Build the image (you can tag it as you like)
sudo docker build -t trademark_backend:latest .

# Run the container (adjust ports or environment variables as needed)
sudo docker run -d \
    --name trademark_backend \
    -p 8000:8000 \
    trademark_backend:latest
```

The API will now be accessible at `http://<EC2_PUBLIC_IP>:8000`.

### 6. (Optional) Run the container with `systemd` for auto‑restart
Create a service file:
```bash
sudo tee /etc/systemd/system/trademark_backend.service > /dev/null <<'EOF'
[Unit]
Description=Trademark Backend Service
After=network.target docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker run --rm -p 8000:8000 trademark_backend:latest
ExecStop=/usr/bin/docker stop trademark_backend

[Install]
WantedBy=multi-user.target
EOF
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trademark_backend.service
sudo systemctl start trademark_backend.service
```

### 7. Verify the deployment
```bash
curl http://localhost:8000/health  # Adjust endpoint as defined in your API
```
You should receive a response indicating the service is healthy.

---
*Feel free to adjust environment variables, mount volumes, or configure a reverse proxy (e.g., Nginx) as needed for production deployments.*