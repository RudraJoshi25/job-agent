#!/usr/bin/env bash
# One-time EC2 setup script for Ubuntu 22.04 (t2.micro / t3.small).
# Run as: sudo bash ec2_setup.sh
# After running, place .env at /home/ubuntu/job-agent/frontend/backend/.env

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/YOUR_ORG/job-agent.git}"
REPO_DIR="/home/ubuntu/job-agent"
APP_USER="ubuntu"

echo "=== 1. System update ==="
apt-get update -y && apt-get upgrade -y

echo "=== 2. Install Docker ==="
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker "$APP_USER"
systemctl enable docker
systemctl start docker

echo "=== 3. Install Nginx ==="
apt-get install -y nginx
systemctl enable nginx

echo "=== 4. Install Certbot ==="
apt-get install -y certbot python3-certbot-nginx

echo "=== 5. Configure UFW firewall ==="
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "=== 6. Clone repo ==="
if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$REPO_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$REPO_DIR"
    chown -R "$APP_USER":"$APP_USER" "$REPO_DIR"
fi

echo "=== 7. Configure Nginx ==="
cp "$REPO_DIR/scripts/nginx.conf" /etc/nginx/sites-available/jobpilot
ln -sf /etc/nginx/sites-available/jobpilot /etc/nginx/sites-enabled/jobpilot
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "=== 8. Create systemd service ==="
cat > /etc/systemd/system/jobpilot.service <<EOF
[Unit]
Description=JobPilot Backend
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${REPO_DIR}
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
User=${APP_USER}
Group=${APP_USER}
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable jobpilot

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Copy .env to ${REPO_DIR}/frontend/backend/.env"
echo "  2. Run: systemctl start jobpilot"
echo "  3. For HTTPS: certbot --nginx -d your-domain.com"
