# 🚀 Deployment Guide for Energy Consumption Predictor

This guide provides multiple deployment options for your Energy Consumption Predictor web application.

## 📋 Quick Deployment Options

### 1. 🖥️ **Local Network Deployment (Easiest)**
Run on your local network so others can access it:

```powershell
# Start the app accessible to your network
python run_webapp.py
```

The app will be available at:
- **Your computer**: http://localhost:5000
- **Network access**: http://YOUR_IP_ADDRESS:5000

**Find your IP address:**
```powershell
ipconfig | findstr IPv4
```

### 2. 🌐 **Production-Ready Local Server**
Using Gunicorn (more stable for production):

```powershell
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (Windows alternative: waitress)
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 --call run_webapp:app
```

### 3. ☁️ **Cloud Deployment Options**

## 🚀 Deployment Methods

### Option A: Heroku (Free Tier Available)

1. **Install Heroku CLI**: Download from https://devcenter.heroku.com/articles/heroku-cli
2. **Login**: `heroku login`
3. **Create app**: `heroku create your-energy-predictor`
4. **Deploy**: `git push heroku main`

### Option B: Railway (Modern & Easy)

1. **Visit**: https://railway.app
2. **Connect GitHub repo**
3. **Deploy automatically**

### Option C: Render (Free Tier)

1. **Visit**: https://render.com
2. **Connect GitHub**
3. **Deploy as Web Service**

### Option D: DigitalOcean App Platform

1. **Visit**: https://cloud.digitalocean.com/apps
2. **Create from GitHub**
3. **Deploy automatically**

### Option E: AWS Elastic Beanstalk

1. **Install AWS CLI**
2. **Deploy with EB CLI**

## 📁 Required Files for Deployment

### Procfile (for Heroku)
```
web: waitress-serve --port=$PORT --call run_webapp:app
```

### runtime.txt (for Heroku)
```
python-3.11.9
```

### app.yaml (for Google Cloud)
```yaml
runtime: python311
env: standard
instance_class: F1

automatic_scaling:
  min_instances: 0
  max_instances: 2
```

### Dockerfile (for containerized deployment)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "run_webapp.py"]
```

## 🔧 Environment Configuration

### Environment Variables
Set these for production:
```bash
FLASK_ENV=production
PORT=5000
```

### Security Settings
- Use HTTPS in production
- Set secure headers
- Enable CORS properly
- Use environment variables for secrets

## 📊 Performance Optimization

### For Production:
1. **Use Gunicorn/Waitress** instead of Flask dev server
2. **Enable gzip compression**
3. **Use CDN for static assets**
4. **Add caching headers**
5. **Monitor with logging**

## 🛠️ Monitoring & Maintenance

### Health Checks
The app includes `/health` endpoint for monitoring

### Logging
Logs are configured for production monitoring

### Scaling
- Start with 1 instance
- Scale based on usage
- Monitor CPU/memory usage

## 🔒 Security Considerations

1. **HTTPS**: Always use SSL in production
2. **CORS**: Configure for your domain
3. **Rate Limiting**: Add if needed
4. **Input Validation**: Already implemented
5. **Error Handling**: Production-ready error pages

## 🎯 Recommended Deployment Path

**For beginners**: Start with Railway or Render (easiest)
**For teams**: Use DigitalOcean or AWS
**For enterprises**: Use AWS/Azure/GCP with proper DevOps

## 📞 Support Resources

- Flask deployment docs: https://flask.palletsprojects.com/en/2.3.x/deploying/
- Heroku Python guide: https://devcenter.heroku.com/articles/getting-started-with-python
- Railway docs: https://docs.railway.app/