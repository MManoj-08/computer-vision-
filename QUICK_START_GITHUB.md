# Git Setup and GitHub Upload Commands

## STEP 1: Configure Git (First Time Only)

Run these commands with YOUR information:

```powershell
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your-email@example.com"
```

Example:
```powershell
git config --global user.name "John Doe"
git config --global user.email "john.doe@example.com"
```

---

## STEP 2: Commit Your Changes

```powershell
git commit -m "Initial commit: Noise reduction case study with linear filters"
```

---

## STEP 3: Create GitHub Repository

1. Go to: **https://github.com/new**
2. Repository name: `noise-reduction-case-study`
3. Description: `Industry case study: Noise reduction in camera images using linear filters`
4. Choose **Public** (or Private if you prefer)
5. **DO NOT** check "Initialize with README" (we already have one)
6. Click **"Create repository"**

---

## STEP 4: Link and Push to GitHub

Replace `YOUR-USERNAME` with your GitHub username:

```powershell
# Add remote repository
git remote add origin https://github.com/YOUR-USERNAME/noise-reduction-case-study.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

When prompted, enter your GitHub credentials:
- Username: Your GitHub username
- Password: Your Personal Access Token (not your actual password)

### How to get Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "noise-reduction-project"
4. Select scopes: Check "repo"
5. Click "Generate token"
6. Copy the token (you won't see it again!)
7. Use this token as your password when pushing

---

## Quick Copy-Paste Version

After configuring git (Step 1) and creating GitHub repo (Step 3):

```powershell
# Commit
git commit -m "Initial commit: Noise reduction case study"

# Add remote (REPLACE YOUR-USERNAME!)
git remote add origin https://github.com/YOUR-USERNAME/noise-reduction-case-study.git

# Push
git branch -M main
git push -u origin main
```

---

## DONE! 🎉

Your project will be live at:
`https://github.com/YOUR-USERNAME/noise-reduction-case-study`

---

## Future Updates

When you make changes to your files:

```powershell
git add .
git commit -m "Description of what you changed"
git push
```

---

## Files Already Staged:

✅ industry_case_study.py
✅ camera.jpg
✅ requirements.txt
✅ README.md
✅ .gitignore
✅ GITHUB_UPLOAD_GUIDE.md
✅ output/ (all result images)
✅ Other Python scripts
✅ Jupyter notebook

Everything is ready to push once you:
1. Configure git (your name & email)
2. Create GitHub repository
3. Run the push commands

---

Need help? Check GITHUB_UPLOAD_GUIDE.md for detailed instructions!
