# Add Python 3.10 to PATH permanently
# Run this script in PowerShell AS ADMINISTRATOR

$python310Path = "C:\Users\getma\AppData\Local\Programs\Python\Python310"
$python310Scripts = "C:\Users\getma\AppData\Local\Programs\Python\Python310\Scripts"

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if already in PATH
if ($currentPath -notlike "*$python310Path*") {
    Write-Host "Adding Python 3.10 to PATH..." -ForegroundColor Green
    
    # Add to BEGINNING of PATH (so it takes priority over Python 3.7)
    $newPath = "$python310Path;$python310Scripts;$currentPath"
    
    # Set the new PATH
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    
    Write-Host "✓ Python 3.10 added to PATH!" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: You must close and reopen PowerShell for this to take effect." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After reopening, test with: python --version" -ForegroundColor Cyan
    Write-Host "Should show: Python 3.10.11" -ForegroundColor Cyan
} else {
    Write-Host "Python 3.10 is already in PATH" -ForegroundColor Yellow
}
