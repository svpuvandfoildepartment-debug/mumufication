# 🔄 How to Update Your GitHub Repo

## Problem
You copied files locally but they're not showing up on GitHub.

## Solution
You need to **commit and push** the changes to GitHub.

---

## Step-by-Step Instructions

### 1. Open Terminal in Your Project Folder
```bash
cd /path/to/your/project
# Example: cd ~/mumufication
```

### 2. Check What Files Changed
```bash
git status
```

You should see something like:
```
modified:   ai_stock_screener.py
modified:   ai_stock_screener_multi_user.py
new file:   ai_stock_screener_us_only.py
```

### 3. Add All Changes
```bash
git add .
```

Or add specific files:
```bash
git add ai_stock_screener.py
git add ai_stock_screener_multi_user.py
git add ai_stock_screener_us_only.py
git add README.md
git add requirements.txt
```

### 4. Commit the Changes
```bash
git commit -m "Update: US-only version + multi-user with login + API config in main screen"
```

### 5. Push to GitHub
```bash
git push origin main
```

Or if your main branch is called `master`:
```bash
git push origin master
```

### 6. Verify on GitHub
Go to: https://github.com/svpuvandfoildepartment-debug/mumufication

You should now see:
- ✅ Updated files
- ✅ New commit message
- ✅ Updated timestamps

---

## Quick Copy-Paste Commands

```bash
# Navigate to your project
cd ~/mumufication

# Check status
git status

# Add all files
git add .

# Commit
git commit -m "Update stock screener: US-only + multi-user + better API config"

# Push
git push origin main

# Done!
```

---

## If You Get Errors

### Error: "fatal: not a git repository"
**Fix:** Initialize git first
```bash
git init
git remote add origin https://github.com/svpuvandfoildepartment-debug/mumufication.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Error: "failed to push some refs"
**Fix:** Pull first, then push
```bash
git pull origin main --rebase
git push origin main
```

### Error: "Permission denied (publickey)"
**Fix:** Use HTTPS instead of SSH
```bash
git remote set-url origin https://github.com/svpuvandfoildepartment-debug/mumufication.git
git push origin main
```

### Error: "Your branch is behind"
**Fix:** Pull first
```bash
git pull origin main
git push origin main
```

---

## What Files to Upload

From the files I gave you, upload these:

```
your-project/
├── ai_stock_screener.py                 # Original single-user (US-only)
├── ai_stock_screener_multi_user.py      # NEW multi-user with login
├── ai_stock_screener_us_only.py         # Same as original, just renamed
├── requirements.txt                      # Dependencies
├── README.md                             # Documentation
├── QUICKSTART.md                         # Quick start guide
├── UPGRADE_GUIDE.md                      # Migration guide
├── NEW_UI_LAYOUT.md                      # UI documentation
├── .gitignore                            # Git ignore rules
└── test_multi_user.py                    # Test script
```

**Important:** Do NOT upload the `user_data/` folder (it should be in .gitignore)

---

## Verify Your Upload Worked

1. Go to https://github.com/svpuvandfoildepartment-debug/mumufication
2. Click on `ai_stock_screener_multi_user.py`
3. Check the file size — should be ~125KB
4. Check line count — should be ~3,000 lines
5. Look for "render_main_config_panel" in the code

If you see those, it worked! ✅

---

## Alternative: Upload via GitHub Web Interface

If Git commands are confusing:

1. Go to https://github.com/svpuvandfoildepartment-debug/mumufication
2. Click **"Add file" → "Upload files"**
3. Drag and drop all the `.py` and `.md` files
4. Scroll down and click **"Commit changes"**

Done!

---

## Check Your Deployed App

If you deployed on Streamlit Cloud:

1. Go to https://share.streamlit.io
2. Find your app
3. Click **"Reboot app"** to use the new code
4. Wait 30 seconds
5. Refresh the browser

The changes should now appear.

---

## Summary

**The issue:** Files are on your computer but not on GitHub  
**The fix:** `git add . && git commit -m "Update" && git push`  
**The result:** Files appear on GitHub and Streamlit Cloud picks them up

Without `git push`, your changes stay local only.
