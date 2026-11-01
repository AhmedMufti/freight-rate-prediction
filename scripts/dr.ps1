# Run commands inside the freight-ml Docker image with the project mounted at /work.
#   scripts\dr.ps1 <cli cmd and flags>            -> python -m freight.cli ...
#   scripts\dr.ps1 -Raw <any command and args>    -> run the command as given (pytest, score.py, ...)
# Uses $args (no PowerShell parameter binding) so flags such as -p or -o pass through untouched.
$raw = $false
$rest = @($args)
if ($rest.Count -gt 0 -and $rest[0] -eq '-Raw') { $raw = $true; $rest = $rest[1..($rest.Count - 1)] }
if ($raw) { $cmd = $rest } else { $cmd = @('python', '-m', 'freight.cli') + $rest }
docker run --rm -v "${PWD}:/work" -w /work -e PYTHONUTF8=1 -e PYTHONPATH=/work/src freight-ml @cmd
exit $LASTEXITCODE
