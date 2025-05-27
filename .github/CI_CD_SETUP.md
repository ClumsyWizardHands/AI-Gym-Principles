# CI/CD Pipeline Setup Guide

This guide will walk you through setting up the complete CI/CD pipeline for the AI Principles Gym project.

## Prerequisites

- GitHub repository at https://github.com/ClumsyWizardHands/AI-Gym-Principles
- Docker Hub account (for container registry)
- Staging and production servers with SSH access
- Codecov.io account (optional, for coverage reports)

## Step 1: Configure GitHub Secrets

Navigate to your repository settings: https://github.com/ClumsyWizardHands/AI-Gym-Principles/settings/secrets/actions

Add the following secrets:

### Docker Hub Credentials
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password or access token
  - To create an access token: https://hub.docker.com/settings/security
  - Use tokens instead of passwords for better security

### Staging Server Credentials
- `STAGING_HOST`: IP address or hostname of your staging server
- `STAGING_USER`: SSH username for staging server (e.g., `ubuntu`, `deploy`)
- `STAGING_SSH_KEY`: Private SSH key for staging server access
  ```bash
  # Generate SSH key pair if needed:
  ssh-keygen -t ed25519 -C "github-actions@ai-principles-gym"
  # Copy the private key content to this secret
  # Add the public key to ~/.ssh/authorized_keys on staging server
  ```

### Production Server Credentials
- `PROD_HOST`: IP address or hostname of your production server
- `PROD_USER`: SSH username for production server
- `PROD_SSH_KEY`: Private SSH key for production server access (use a different key than staging)

### Code Coverage (Optional)
- `CODECOV_TOKEN`: Token from codecov.io
  - Sign up at https://codecov.io
  - Add your repository
  - Copy the upload token from the settings

## Step 2: Configure Branch Protection Rules

1. Go to Settings → Branches: https://github.com/ClumsyWizardHands/AI-Gym-Principles/settings/branches

2. Add rule for `main` branch:
   - Branch name pattern: `main`
   - ✅ Require a pull request before merging
     - ✅ Require approvals: 1
     - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
     - ✅ Require branches to be up to date before merging
     - Status checks:
       - `Test Suite`
       - `Security Vulnerability Scan`
       - `CodeQL Analysis`
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

3. Add rule for `develop` branch:
   - Branch name pattern: `develop`
   - ✅ Require status checks to pass before merging
     - Status checks:
       - `Test Suite`
   - ✅ Allow force pushes (for admins only)

## Step 3: Configure Environments

1. Go to Settings → Environments: https://github.com/ClumsyWizardHands/AI-Gym-Principles/settings/environments

2. Create `staging` environment:
   - No required reviewers
   - Add environment secrets if different from repository secrets

3. Create `production` environment:
   - ✅ Required reviewers: Add yourself or team members
   - ✅ Wait timer: 5 minutes (optional)
   - Environment URL: `https://api.principles-gym.ai` (update with your domain)

## Step 4: Server Setup

### On both staging and production servers:

1. Install Docker and Docker Compose:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. Create deployment directory:
   ```bash
   sudo mkdir -p /opt/ai-principles-gym
   sudo chown $USER:$USER /opt/ai-principles-gym
   ```

3. Clone repository:
   ```bash
   cd /opt/ai-principles-gym
   git clone https://github.com/ClumsyWizardHands/AI-Gym-Principles.git .
   ```

4. Create environment file:
   ```bash
   cp deployment/.env.production .env
   # Edit .env with your specific values
   ```

5. Add GitHub Actions SSH key:
   ```bash
   # Add the public key to authorized_keys
   echo "YOUR_PUBLIC_KEY" >> ~/.ssh/authorized_keys
   ```

## Step 5: Configure DNS (Production Only)

1. Point your domain to your production server IP
2. Configure SSL with Let's Encrypt:
   ```bash
   # Install certbot
   sudo apt-get update
   sudo apt-get install certbot python3-certbot-nginx
   
   # Get certificate
   sudo certbot --nginx -d api.principles-gym.ai
   ```

## Step 6: First Deployment

1. Push to `develop` branch to trigger staging deployment:
   ```bash
   git checkout develop
   git push origin develop
   ```

2. Create a PR from `develop` to `main` and merge to trigger production deployment

## Step 7: Monitor Deployments

- Check Actions tab: https://github.com/ClumsyWizardHands/AI-Gym-Principles/actions
- View deployment history: https://github.com/ClumsyWizardHands/AI-Gym-Principles/deployments

## Workflow Features

### CI/CD Pipeline (`ci-cd.yml`)
- **Test Job**: Runs on all PRs and pushes
  - Linting with flake8 and black
  - Type checking with mypy
  - Unit tests with pytest
  - Coverage reports to Codecov
- **Build Job**: Builds and pushes Docker images
  - Multi-platform builds (amd64, arm64)
  - Semantic versioning tags
  - Build cache optimization
- **Deploy Jobs**:
  - Staging: Auto-deploy from `develop`
  - Production: Manual approval required, deploys from `main`
  - Database backups before production deployment
  - Rolling updates with health checks

### Security Scanning (`security.yml`)
- **Weekly scans** plus on every push
- **Dependency scanning** with Safety and pip-audit
- **SAST** with Bandit and CodeQL
- **Container scanning** with Trivy
- **Secret detection** with TruffleHog and Gitleaks
- **License compliance** checking

### Dependency Updates (`dependabot.yml`)
- **Automated PRs** for dependency updates
- **Grouped updates** by category
- **Security updates** prioritized
- **Version constraints** respected

## Troubleshooting

### SSH Connection Issues
```bash
# Test SSH connection
ssh -i /path/to/key username@host
# Check SSH key permissions
chmod 600 ~/.ssh/id_ed25519
```

### Docker Build Failures
```bash
# Clear Docker cache
docker system prune -a
# Check Docker daemon logs
sudo journalctl -u docker
```

### Health Check Failures
```bash
# Check service logs
docker-compose logs api
# Verify service is running
docker-compose ps
```

## Security Best Practices

1. **Rotate secrets regularly** (every 90 days)
2. **Use different SSH keys** for staging and production
3. **Enable 2FA** on GitHub and Docker Hub
4. **Review security scan results** weekly
5. **Keep dependencies updated** via Dependabot

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Access Tokens](https://docs.docker.com/docker-hub/access-tokens/)
- [Codecov Setup](https://docs.codecov.com/docs/quick-start)
- [GitHub Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments)
