# Deployment Guide: Room Acoustics Web App

## Overview
You have 4 main deployment options, ranging from simplest (local) to most sophisticated (cloud):

---

## Option 1: Streamlit Cloud (RECOMMENDED) ⭐
**Complexity:** ⭐ Easy  
**Cost:** FREE  
**Setup time:** 15 minutes  
**User experience:** Colleagues just visit a URL - no installation needed

### Pros:
- ✅ Completely free (unlimited public apps)
- ✅ Zero server management
- ✅ Automatic updates when you push to GitHub
- ✅ SSL/HTTPS by default
- ✅ Works on any device (desktop, tablet, phone)
- ✅ No installation needed for users

### Cons:
- ⚠️ Public by default (can be password-protected with Streamlit auth)
- ⚠️ Limited to 1 GB RAM (usually fine for acoustic analysis)
- ⚠️ Apps sleep after inactivity (takes 10-30s to wake up)

### Setup Steps:

1. **Create GitHub repository**
   ```bash
   # On your computer
   git init acoustic-analyzer
   cd acoustic-analyzer
   
   # Add your files
   cp acoustic_analysis_app.py .
   
   # Create requirements.txt
   cat > requirements.txt << EOF
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyfar>=0.6.0
pyrato>=0.4.0
scipy>=1.10.0
EOF
   
   # Commit and push
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/acoustic-analyzer.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `acoustic_analysis_app.py`
   - Click "Deploy"
   - Wait 2-5 minutes

3. **Share with colleagues**
   - You get a URL like: `https://your-app.streamlit.app`
   - Send this link to colleagues
   - They just click and use - no installation!

### Optional: Add password protection
```python
# Add to top of acoustic_analysis_app.py
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "your_secret_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# Then wrap your main app:
if check_password():
    main()  # Your app code
```

---

## Option 2: Local Network Deployment (Internal Server)
**Complexity:** ⭐⭐ Moderate  
**Cost:** FREE (use existing office computer/server)  
**Setup time:** 30 minutes  
**User experience:** Colleagues visit internal URL (e.g., http://acoustics-server.local:8501)

### Best for:
- Companies with security concerns
- No internet access requirements
- Already have an internal server

### Setup Steps:

1. **Choose a server machine** (can be any Windows/Mac/Linux computer)

2. **Install Python and dependencies**
   ```bash
   # Install Python 3.10 or higher
   # Then install dependencies:
   pip install streamlit pyfar pyrato numpy pandas matplotlib scipy
   ```

3. **Create a startup script**
   
   **Windows (run_app.bat):**
   ```batch
   @echo off
   cd /d C:\path\to\acoustic-analyzer
   streamlit run acoustic_analysis_app.py --server.address 0.0.0.0 --server.port 8501
   pause
   ```
   
   **Mac/Linux (run_app.sh):**
   ```bash
   #!/bin/bash
   cd /path/to/acoustic-analyzer
   streamlit run acoustic_analysis_app.py --server.address 0.0.0.0 --server.port 8501
   ```

4. **Make it auto-start** (optional)
   
   **Windows:** Use Task Scheduler to run on login  
   **Mac:** Use Automator or launchd  
   **Linux:** Create systemd service

5. **Share the URL**
   - Find server IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
   - Share URL: `http://192.168.x.x:8501` with colleagues
   - Optionally set up DNS name in your office network

### Windows Service Example:
```python
# install_service.py
import win32serviceutil
import win32service
import subprocess
import os

class AcousticAnalyzerService(win32serviceutil.ServiceFramework):
    _svc_name_ = "AcousticAnalyzer"
    _svc_display_name_ = "Room Acoustics Analyzer"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.process = None
    
    def SvcDoRun(self):
        self.process = subprocess.Popen([
            'streamlit', 'run', 
            'C:\\path\\to\\acoustic_analysis_app.py',
            '--server.address', '0.0.0.0',
            '--server.port', '8501'
        ])
        self.process.wait()
    
    def SvcStop(self):
        if self.process:
            self.process.terminate()

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(AcousticAnalyzerService)
```

---

## Option 3: Docker Container (Cross-platform)
**Complexity:** ⭐⭐⭐ Advanced  
**Cost:** FREE  
**Setup time:** 45 minutes  
**User experience:** Same as Option 2, but more reliable

### Best for:
- IT departments comfortable with Docker
- Consistent deployment across environments
- Easy scaling and updating

### Setup Files:

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY acoustic_analysis_app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "acoustic_analysis_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  acoustic-analyzer:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
    volumes:
      - ./data:/app/data  # Optional: persist data
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

**Deploy:**
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Update
git pull
docker-compose up -d --build
```

---

## Option 4: Cloud Hosting (AWS/Azure/GCP)
**Complexity:** ⭐⭐⭐⭐ Expert  
**Cost:** ~$5-20/month  
**Setup time:** 1-2 hours  
**User experience:** Professional URL, fast, always on

### Best for:
- Large teams
- Need 24/7 availability
- Custom domain (e.g., acoustics.yourcompany.com)
- More than 1GB RAM needed

### AWS Lightsail Example (Simplest Cloud Option):

1. **Create Lightsail instance** ($3.50/month)
   - OS: Ubuntu 22.04
   - Plan: $3.50 or $5/month
   - Open port 8501 in firewall

2. **SSH and setup**:
   ```bash
   ssh ubuntu@your-instance-ip
   
   # Install Python
   sudo apt update
   sudo apt install python3-pip python3-venv libsndfile1 -y
   
   # Create app directory
   mkdir acoustic-analyzer
   cd acoustic-analyzer
   
   # Upload your files (use scp or git)
   git clone https://github.com/YOUR_USERNAME/acoustic-analyzer.git .
   
   # Install dependencies
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Run with PM2 for auto-restart
   sudo npm install -g pm2
   pm2 start "streamlit run acoustic_analysis_app.py --server.address 0.0.0.0 --server.port 8501" --name acoustic
   pm2 save
   pm2 startup
   ```

3. **Add domain** (optional):
   - Point DNS to instance IP
   - Set up nginx reverse proxy
   - Add SSL with Let's Encrypt

---

## Comparison Table

| Feature | Streamlit Cloud | Local Server | Docker | Cloud (AWS) |
|---------|----------------|--------------|--------|-------------|
| **Setup Time** | 15 min | 30 min | 45 min | 2 hours |
| **Cost** | FREE | FREE | FREE | $5-20/mo |
| **Maintenance** | None | Low | Low | Medium |
| **Reliability** | Medium | Medium | High | High |
| **Security** | Public/Password | Private | Private | Configurable |
| **Performance** | Good | Excellent | Excellent | Excellent |
| **Scalability** | Limited | Low | Medium | High |
| **Best for** | Quick start | Small teams | IT teams | Large orgs |

---

## My Recommendation for You

**Start with Streamlit Cloud (Option 1)**

Reasons:
1. ✅ **Zero setup for colleagues** - just send them a link
2. ✅ **Free forever** for your use case
3. ✅ **5-minute setup** - you can have it running today
4. ✅ **Professional appearance** - looks like a real product
5. ✅ **Automatic updates** - push to GitHub, app updates automatically
6. ✅ **No maintenance** - Streamlit handles all infrastructure

Then, IF you need:
- **More privacy**: Move to Option 2 (Local Server)
- **More reliability**: Move to Option 3 (Docker on local server)
- **Public-facing with custom domain**: Move to Option 4 (Cloud)

---

## Quick Start: Get Running in 5 Minutes

```bash
# 1. Create folder
mkdir acoustic-analyzer
cd acoustic-analyzer

# 2. Copy the app file
# (Use the acoustic_analysis_app.py from outputs)

# 3. Create requirements.txt
echo "streamlit>=1.28.0
pyfar>=0.6.0
pyrato>=0.4.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0" > requirements.txt

# 4. Install and run locally first (test)
pip install -r requirements.txt
streamlit run acoustic_analysis_app.py

# 5. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git push to GitHub

# 6. Deploy to Streamlit Cloud
# - Go to share.streamlit.io
# - Connect GitHub repo
# - Deploy!
```

---

## Troubleshooting

### "ModuleNotFoundError"
- Make sure requirements.txt includes all dependencies
- Try: `pip install -r requirements.txt --upgrade`

### "Port already in use"
- Change port: `streamlit run app.py --server.port 8502`

### "App runs slow with large files"
- Consider upgrading to paid Streamlit tier ($20/month for 4GB RAM)
- Or deploy to cloud with more resources

### "Colleagues can't access"
- Check firewall settings
- Verify network connectivity
- Make sure server address is `0.0.0.0` not `localhost`

---

## Security Best Practices

1. **Don't commit secrets** - use Streamlit secrets management
2. **Add password protection** if deploying publicly
3. **Limit file upload size** - default 200MB is usually fine
4. **Use HTTPS** - Streamlit Cloud provides this automatically
5. **Regular updates** - keep dependencies updated

---

## Need Help?

- Streamlit docs: https://docs.streamlit.io
- Streamlit forum: https://discuss.streamlit.io
- Docker docs: https://docs.docker.com

Feel free to ask if you need help with any deployment option!
