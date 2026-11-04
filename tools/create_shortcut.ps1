# Creates "AuNW Analysis" shortcuts (desktop + start menu) with the beaker icon.
# Called by install_windows.bat; can also be run on its own.
$root   = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root ".venv\Scripts\pythonw.exe"
$icon   = Join-Path $root "assets\aunw.ico"
$places = @([Environment]::GetFolderPath("Desktop"),
            (Join-Path ([Environment]::GetFolderPath("Programs")) ""))
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in $places) {
    $lnk = $shell.CreateShortcut((Join-Path $dir "AuNW Analysis.lnk"))
    $lnk.TargetPath       = $target
    $lnk.Arguments        = '"' + (Join-Path $root "desktop.py") + '"'
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation     = $icon
    $lnk.Description      = "AuNW Analysis - length and width of gold nanowires in SEM images"
    $lnk.Save()
    Write-Host "Shortcut created in $dir"
}
