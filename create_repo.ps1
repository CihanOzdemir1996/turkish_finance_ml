# Script to create GitHub repository and push code
# Make sure you have a GitHub Personal Access Token with 'repo' scope

$repoName = "turkish_finance_ml"
$username = "CihanOzdemir1996"
$description = "BIST-100 AI Prediction System - Machine learning models for Turkish stock market price direction prediction"

Write-Host "Creating GitHub repository: $repoName" -ForegroundColor Green

# Check if token is provided
$token = $env:GITHUB_TOKEN
if (-not $token) {
    Write-Host "`nGitHub Personal Access Token not found in environment." -ForegroundColor Yellow
    Write-Host "Please provide your GitHub token:" -ForegroundColor Yellow
    Write-Host "1. Get token from: https://github.com/settings/tokens" -ForegroundColor Cyan
    Write-Host "2. Run: `$env:GITHUB_TOKEN='your_token_here'`" -ForegroundColor Cyan
    Write-Host "3. Then run this script again" -ForegroundColor Cyan
    exit 1
}

# Create repository via GitHub API
$headers = @{
    'Authorization' = "token $token"
    'Accept' = 'application/vnd.github.v3+json'
}

$body = @{
    name = $repoName
    description = $description
    private = $false
} | ConvertTo-Json

try {
    Write-Host "Creating repository..." -ForegroundColor Yellow
    $response = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Headers $headers -Method Post -Body $body -ContentType 'application/json'
    Write-Host "Repository created successfully!" -ForegroundColor Green
    Write-Host "Repository URL: $($response.html_url)" -ForegroundColor Cyan
    
    # Push code
    Write-Host "`nPushing code to repository..." -ForegroundColor Yellow
    git push -u origin main
    Write-Host "`nCode pushed successfully!" -ForegroundColor Green
    Write-Host "View your repository at: $($response.html_url)" -ForegroundColor Cyan
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "Authentication failed. Please check your token." -ForegroundColor Red
    } elseif ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "Repository might already exist or name is invalid." -ForegroundColor Red
    }
}
