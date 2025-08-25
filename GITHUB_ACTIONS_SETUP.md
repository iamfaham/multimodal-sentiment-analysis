# GitHub Actions Setup for Hugging Face Space Deployment

This guide will help you set up automatic deployment to a Hugging Face Space whenever you push to your GitHub repository.

## 🚀 What This Workflow Does

The GitHub Actions workflow (`.github/workflows/deploy-to-hf-space.yml`) will:

1. **Trigger automatically** when you push to `main` or `master` branch
2. **Test your application** by importing key dependencies
3. **Create a Hugging Face Space** if it doesn't exist
4. **Upload your code** to the space
5. **Deploy your Streamlit app** automatically

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **Hugging Face Account**: You need a Hugging Face account
3. **Hugging Face Access Token**: You need to generate an access token

## 🔑 Step 1: Get Your Hugging Face Access Token

1. Go to [Hugging Face](https://huggingface.co/) and sign in
2. Click on your profile picture → **Settings**
3. Go to **Access Tokens** in the left sidebar
4. Click **New token**
5. Give it a name (e.g., "GitHub Actions")
6. Select **Write** permissions
7. Click **Generate token**
8. **Copy the token** (you won't see it again!)

## 🔧 Step 2: Set Up GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** tab
3. Click **Secrets and variables** → **Actions** in the left sidebar
4. Click **New repository secret**
5. Add these three secrets:

### Secret 1: HF_TOKEN

- **Name**: `HF_TOKEN`
- **Value**: Your Hugging Face access token from Step 1

### Secret 2: HF_USERNAME

- **Name**: `HF_USERNAME`
- **Value**: Your Hugging Face username (not email)

### Secret 3: SPACE_NAME

- **Name**: `SPACE_NAME`
- **Value**: The name you want for your space (e.g., `sentiment-fused`)

## 📁 Step 3: Verify File Structure

Make sure these files exist in your repository root:

- ✅ `app.py` - Your main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Project documentation
- ✅ `simple_model_manager.py` - Model management utilities

## 🚀 Step 4: Push to GitHub

1. Commit and push your changes to the `main` or `master` branch:

   ```bash
   git add .
   git commit -m "Add GitHub Actions workflow for HF Space deployment"
   git push origin main
   ```

2. The workflow will automatically trigger and deploy to your Hugging Face Space!

## 📊 Step 5: Monitor Deployment

1. Go to your GitHub repository
2. Click **Actions** tab
3. You'll see the "Deploy to Hugging Face Space" workflow running
4. Click on it to see detailed logs
5. Wait for it to complete (usually 2-5 minutes)

## 🌐 Step 6: Access Your Deployed App

Once deployment is successful, your app will be available at:

```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Workflow not triggering"

- Ensure you're pushing to `main` or `master` branch
- Check that the workflow file is in `.github/workflows/` directory
- Verify the file has `.yml` extension

#### 2. "Authentication failed"

- Double-check your `HF_TOKEN` secret
- Ensure the token has **Write** permissions
- Verify your `HF_USERNAME` is correct (username, not email)

#### 3. "Space creation failed"

- Check if a space with that name already exists
- Ensure your Hugging Face account is verified
- Check the workflow logs for specific error messages

#### 4. "File upload failed"

- Verify all required files exist in your repository
- Check file permissions and names
- Ensure files aren't too large

### Debug Steps

1. **Check workflow logs**: Go to Actions → Workflow → Job → Step
2. **Verify secrets**: Go to Settings → Secrets and variables → Actions
3. **Test locally**: Try running `python -c "import huggingface_hub"`
4. **Check HF account**: Ensure your Hugging Face account is active

## 🔄 Automatic Updates

After initial setup, **every push to main/master will automatically update your Hugging Face Space**:

1. Make changes to your code
2. Commit and push to GitHub
3. GitHub Actions automatically deploys to HF Space
4. Your app updates in real-time!

## 📝 Customization

### Change Trigger Branches

Edit the workflow file to trigger on different branches:

```yaml
on:
  push:
    branches: [main, develop, feature-branch]
```

### Add More Files

Modify the `files_to_upload` list in the workflow:

```python
files_to_upload = [
    'app.py',
    'requirements.txt',
    'README.md',
    'simple_model_manager.py',
    'your_new_file.py'  # Add more files here
]
```

### Change Space Hardware

Modify the space creation to use different hardware:

```python
api.create_space(
    repo_id='$HF_USERNAME/$SPACE_NAME',
    space_sdk='streamlit',
    space_hardware='gpu-t4'  # or 'cpu-basic', 'gpu-a10g'
)
```

## 🎉 Success!

Once everything is set up, you'll have:

- ✅ **Automatic deployment** on every push
- ✅ **Real-time updates** to your HF Space
- ✅ **Professional hosting** for your Streamlit app
- ✅ **Zero manual deployment** required

Your sentiment analysis app will be live and automatically updated whenever you push changes to GitHub!

---

**Need help?** Check the workflow logs in the Actions tab or open an issue in your repository.
