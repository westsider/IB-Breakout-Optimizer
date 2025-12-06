$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\IB Breakout Optimizer.lnk")
$Shortcut.TargetPath = "C:\Users\Warren\Projects\ib_breakout_optimizer\run_optimizer.bat"
$Shortcut.WorkingDirectory = "C:\Users\Warren\Projects\ib_breakout_optimizer"
$Shortcut.WindowStyle = 7
$Shortcut.Save()
Write-Host "Desktop shortcut created!"
