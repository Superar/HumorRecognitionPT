. .\scripts\experiments\cross_corpus\0-variables.ps1

Write-Host 'HEADLINES -> ONE-LINERS' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\headlines_oneliners_content_features

Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\headlines_oneliners_humor_features

Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\headlines_oneliners_all_features

Write-Host 'ONE-LINERS -> HEADLINES' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n

Write-Host 'CONTENT FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\oneliners_headlines_content_features

Write-Host 'HUMOR FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\oneliners_headlines_humor_features

Write-Host 'ALL FEATURES' -ForegroundColor DarkGreen -BackgroundColor Red -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\oneliners_headlines_all_features

Write-Host 'TRANSFORMER' -ForegroundColor DarkGreen -BackgroundColor Cyan -NoNewline
Write-Output `n
python .\scripts\experiments\Clemencio2019\6-evaluation.py .\results\predictions\transformers