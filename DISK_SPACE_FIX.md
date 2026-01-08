# Fixing "No space left on device" Error

## Problem
Your C: drive is completely full (0 bytes free). This prevents Git from creating objects.

## Solution Steps

### Step 1: Free Up Disk Space

You need to free at least 1-2 GB of space. Here are some options:

#### Option A: Clean Windows Temp Files
```powershell
# Run Disk Cleanup (GUI)
cleanmgr

# Or clean temp files directly
Remove-Item $env:TEMP\* -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item C:\Windows\Temp\* -Recurse -Force -ErrorAction SilentlyContinue
```

#### Option B: Remove Large Files from Project
The following files are large and excluded from git (but still taking disk space):
- `hybrid_data.pt` (210 MB)
- `processed_data_2d_labeled.pt` (205 MB)
- `hard_model.pth` (size unknown)

You can temporarily move these to an external drive or delete them if you can regenerate them.

```powershell
# Check current directory size
Get-ChildItem -Recurse | Measure-Object -Property Length -Sum | Select-Object @{Name="Size(GB)";Expression={[math]::Round($_.Sum/1GB,2)}}

# Move large files to external drive (example)
# Move-Item hybrid_data.pt D:\Backup\
```

#### Option C: Clean Git Objects (Already Done)
We've already cleaned Git objects. Check if space was freed.

### Step 2: Verify Free Space

```powershell
Get-PSDrive C | Select-Object Used,Free,@{Name="Free(GB)";Expression={[math]::Round($_.Free/1GB,2)}}
```

**You need at least 500 MB free** to proceed with git operations.

### Step 3: Once Space is Available

After freeing space, continue with GitHub upload:

```powershell
# 1. Add files (large .pt files will be excluded by .gitignore)
git add .

# 2. Commit
git commit -m "Initial commit: Music Clustering with VAE project"

# 3. Create repository on GitHub (via website: https://github.com/new)

# 4. Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/MusicClusteringProject.git

# 5. Rename branch to main
git branch -M main

# 6. Push
git push -u origin main
```

## Quick Check: What's Using Disk Space?

```powershell
# Check largest directories
Get-ChildItem C:\ -Directory -ErrorAction SilentlyContinue | 
    ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue | 
                 Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        [PSCustomObject]@{
            Path = $_.FullName
            SizeGB = [math]::Round($size/1GB, 2)
        }
    } | Sort-Object SizeGB -Descending | Select-Object -First 10
```

## Alternative: Use External Drive

If you can't free space on C:, you could:
1. Move the entire project to another drive (D:, E:, etc.)
2. Initialize git there
3. Push from there

```powershell
# Example: Move project to D: drive
Move-Item C:\Users\sanch\MusicClusteringProject D:\Projects\
cd D:\Projects\MusicClusteringProject
git init
git add .
git commit -m "Initial commit"
```

