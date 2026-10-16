@echo off
setlocal

:: Ask user for YouTube URL
set /p yturl=Enter YouTube video URL: 

:: Download best 720p video + best audio, and name it as "title.ext"
yt-dlp -f "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]" --merge-output-format mp4 -o "%%(title)s.%%(ext)s" "%yturl%"

pause
