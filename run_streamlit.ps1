# Tree Health Monitor - Streamlit Launch Script
# This script activates the virtual environment and starts the Streamlit application

Write-Host "🌳 Starting Urban Tree Health Monitor..." -ForegroundColor Green
Write-Host "Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Virtual environment activated successfully!" -ForegroundColor Green
    Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
    
    # Start Streamlit
    streamlit run streamlit_app.py
} else {
    Write-Host "❌ Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host "Please make sure the virtual environment exists in the .\venv\ directory" -ForegroundColor Yellow
    pause
}