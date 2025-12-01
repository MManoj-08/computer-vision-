# GitHub Upload Guide

Follow these steps to upload this project to GitHub:

## Option 1: Using GitHub Desktop (Easiest)

1. **Download GitHub Desktop** (if not installed):
   - Go to: https://desktop.github.com/
   - Install and sign in with your GitHub account

2. **Create Repository**:
   - Open GitHub Desktop
   - Click "File" → "Add Local Repository"
   - Choose this folder: `C:\Users\manu9\Desktop\cv1`
   - Click "Create Repository" if prompted

3. **Make Initial Commit**:
   - You'll see all files listed
   - Write commit message: "Initial commit: Noise reduction case study"
   - Click "Commit to main"

4. **Publish to GitHub**:
   - Click "Publish repository"
   - Name: `noise-reduction-case-study` (or your choice)
   - Description: "Industry case study on noise reduction using linear filters"
   - Uncheck "Keep this code private" if you want it public
   - Click "Publish repository"

✅ Done! Your project is now on GitHub!

---

## Option 2: Using Command Line (Git)

### Step 1: Initialize Git Repository

```powershell
# Navigate to project folder (if not already there)
cd C:\Users\manu9\Desktop\cv1

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Noise reduction industry case study"
```

### Step 2: Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `noise-reduction-case-study`
3. Description: `Industry case study: Noise reduction in camera images using linear filters (Samsung/Apple)`
4. Choose Public or Private
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 3: Push to GitHub

GitHub will show you commands. Copy your repository URL and run:

```powershell
# Add remote repository (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/noise-reduction-case-study.git

# Push to GitHub
git branch -M main
git push -u origin main
```

Enter your GitHub credentials when prompted.

✅ Done! Your project is now on GitHub!

---

## Option 3: Using GitHub Web Interface (Manual Upload)

### Step 1: Create Repository
1. Go to: https://github.com/new
2. Repository name: `noise-reduction-case-study`
3. Description: `Industry case study: Noise reduction in camera images`
4. Choose Public or Private
5. Click "Create repository"

### Step 2: Upload Files
1. On your new repository page, click "uploading an existing file"
2. Drag and drop all files from `C:\Users\manu9\Desktop\cv1`:
   - `industry_case_study.py`
   - `camera.jpg`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `output/` folder (all PNG files)
3. Commit message: "Initial commit: Add noise reduction case study"
4. Click "Commit changes"

✅ Done! Your project is now on GitHub!

---

## What Files to Upload

### Essential Files:
- ✅ `industry_case_study.py` - Main script
- ✅ `camera.jpg` - Your input image
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation
- ✅ `.gitignore` - Git ignore rules

### Optional (Output folder):
- ✅ `output/comparison_results.png` - Main result (recommended for README preview)
- ⚠️ Other output PNGs (optional, can be regenerated)

---

## After Uploading

### View Your Project
Your project will be at: `https://github.com/USERNAME/noise-reduction-case-study`

### Share Your Work
Share the link with:
- Classmates
- Instructors
- On LinkedIn (as a project showcase)
- In your portfolio

### Keep It Updated
To update files later:

**Using GitHub Desktop:**
1. Make changes to files
2. Open GitHub Desktop
3. Write commit message
4. Click "Commit to main"
5. Click "Push origin"

**Using Command Line:**
```powershell
git add .
git commit -m "Update: description of changes"
git push
```

---

## Recommended Repository Settings

### Add Topics (for discoverability):
- `computer-vision`
- `image-processing`
- `opencv`
- `python`
- `noise-reduction`
- `linear-filters`
- `convolution`
- `case-study`

### Add README Preview:
Make sure `output/comparison_results.png` is uploaded so the image shows in README.

---

## Troubleshooting

### Issue: "File too large"
- GitHub has 100MB file limit
- Compress large images if needed
- Use Git LFS for large files

### Issue: "Permission denied"
- Check GitHub login credentials
- Use Personal Access Token instead of password
- Generate token at: https://github.com/settings/tokens

### Issue: "Remote already exists"
```powershell
git remote remove origin
git remote add origin YOUR_REPO_URL
```

---

## Need Help?

- GitHub Documentation: https://docs.github.com
- GitHub Learning Lab: https://lab.github.com
- GitHub Desktop Help: https://docs.github.com/en/desktop

---

Good luck with your GitHub upload! 🚀
