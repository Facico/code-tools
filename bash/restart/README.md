把restart.sh放到/etc/systemd/system/里面
```bash
sudo systemctl daemon-reload
sudo systemctl restart restart.service
sudo systemctl status restart.service
sudo systemctl enable restart.service #开机自启
sudo journalctl -xe #查看系统日志
```