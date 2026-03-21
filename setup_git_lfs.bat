@echo off
setlocal enabledelayedexpansion

echo [INFO] Setting up Git LFS for F1 project
where git >nul 2>nul || (echo [FAIL] Git not found & exit /b 1)
where git-lfs >nul 2>nul || (
  echo [FAIL] git-lfs not found. Install from https://git-lfs.github.com/
  exit /b 1
)

git lfs install || (echo [FAIL] git lfs install failed & exit /b 1)

git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.ckpt"
git lfs track "*.h5"
git lfs track "*.hdf5"
git lfs track "*.npz"
git lfs track "*.mp4"
git lfs track "*.avi"
git lfs track "*.mov"
git lfs track "*.kn5"
git lfs track "*.knh"

echo [OK] Git LFS tracking configured.
echo [INFO] Commit .gitattributes and push.
exit /b 0
