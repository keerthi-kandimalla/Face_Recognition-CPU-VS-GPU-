import urllib.request
import zipfile
import os
import sys

base_dir = r"C:\Users\Keerthi Sree\AppData\Local\Programs"
git_dir = os.path.join(base_dir, "Git")
gh_dir = os.path.join(base_dir, "gh")
os.makedirs(git_dir, exist_ok=True)
os.makedirs(gh_dir, exist_ok=True)

mingit_url = "https://github.com/git-for-windows/git/releases/download/v2.56.1.windows.1/MinGit-2.56.1-64-bit.zip"
gh_url = "https://github.com/cli/cli/releases/download/v2.103.3/gh_2.103.3_windows_amd64.zip"

headers = {'User-Agent': 'Mozilla/5.0'}

# 1. Download and extract MinGit
mingit_zip = os.path.join(base_dir, "mingit.zip")
print("Downloading MinGit...")
req = urllib.request.Request(mingit_url, headers=headers)
with urllib.request.urlopen(req) as resp, open(mingit_zip, 'wb') as out_f:
    out_f.write(resp.read())

print("Extracting MinGit...")
with zipfile.ZipFile(mingit_zip, 'r') as z:
    z.extractall(git_dir)
os.remove(mingit_zip)
print("MinGit extracted successfully.")

# 2. Download and extract GH CLI
gh_zip = os.path.join(base_dir, "gh.zip")
print("Downloading GH CLI...")
req = urllib.request.Request(gh_url, headers=headers)
with urllib.request.urlopen(req) as resp, open(gh_zip, 'wb') as out_f:
    out_f.write(resp.read())

print("Extracting GH CLI...")
with zipfile.ZipFile(gh_zip, 'r') as z:
    z.extractall(gh_dir)
os.remove(gh_zip)
print("GH CLI extracted successfully.")

# Check files
git_cmd = os.path.join(git_dir, "cmd", "git.exe")
print("git.exe exists:", os.path.exists(git_cmd))
