@echo off
REM 毎週月曜に自動実行する差分スクレイピング
REM タスクスケジューラに登録して使う

cd /d C:\JRAtool\TEST_action-claude-dreamy-pasteur-jocwqq

echo [%date% %time%] 週次スクレイピング開始 >> logs\scheduler.log
python scripts\collect_recent.py --weeks 2 >> logs\scheduler.log 2>&1
echo [%date% %time%] 週次スクレイピング完了 >> logs\scheduler.log
