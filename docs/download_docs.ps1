# Create docs directory if it doesn't exist
New-Item -ItemType Directory -Force -Path "docs"

# Download main documentation
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/index.html" -OutFile "docs/index.html"
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/datasets.html" -OutFile "docs/datasets.html"
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/models.html" -OutFile "docs/models.html"
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/doctr.io.html" -OutFile "docs/doctr.io.html"
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/doctr.models.html" -OutFile "docs/doctr.models.html"

# Download CSS and other assets
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/_static/sphinx_rtd_theme.css" -OutFile "docs/sphinx_rtd_theme.css"
Invoke-WebRequest -Uri "https://mindee.github.io/doctr/_static/basic.css" -OutFile "docs/basic.css"

Write-Host "Documentation downloaded to docs folder" 