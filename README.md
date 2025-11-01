# spike-clip

## Setup

```bash
# SSH into remote server
ssh -i lamdba_labs_key.pem user@remote_ip

# Clone repository
git clone https://github.com/nikhi3632/spike-clip.git
cd spike-clip

# Setup virtual environment and activate
python3 -m venv venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Fetch data
make data

# Run inference test
make test

# Clean Python cache files
make clean
```