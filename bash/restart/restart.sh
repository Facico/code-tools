echo "v2ray check"
cd /home/fanchenghao/v2ray
echo "当前用户是：$USER"
echo "当前用户是：$(whoami)"
if [ "$resultv2ray" != "" ] && [ $? -eq 0 ]; then
    echo "there is already a tmux named v2ray"
    tmux send-keys -t v2ray "bash v2ray.sh" C-m
else
    echo "new a tmux named v2ray"
    tmux new -d -s v2ray
    tmux send-keys -t v2ray "bash v2ray.sh" C-m
fi

echo "hust-network check"
cd /home/fanchenghao/minieap-master
resultnetwork=$(tmux ls | grep hust-network)
if [ "$resultnetwork" != "" ] && [ $? -eq 0 ]; then
    echo "there is already a tmux named hust-network"
    tmux send-keys -t hust-network "./hust-network-login ./hust_network.conf" C-m
else
    echo "new a tmux named hust-network"
    tmux new -d -s hust-network
    tmux send-keys -t hust-network "./hust-network-login ./hust_network.conf" C-m
fi

echo "frp check"
cd /home/fanchenghao/
resultfrp=$(tmux ls | grep frp)
if [ "$resultfrp" != "" ] && [ $? -eq 0 ]; then
    echo "there is already a tmux named frp"
    tmux send-keys -t frp "./frpc" C-m
else
    echo "new a tmux named frp"
    tmux new -d -s frp
    tmux send-keys -t frp "./frpc" C-m
fi

echo "check mechanical drive"
resulta=$(df -h | grep sda)
resultb=$(df -h | grep sdb)
if [ "$resulta" != "" ]; then
    echo "/dev/sda is already mounted."
else
    echo "/dev/sda is not mounted and will be mounted now."
    sudo mount /dev/sda /data
fi
if [ "$resultb" != "" ]; then
    echo "/dev/sdb is already mounted."
else
    echo "/dev/sdb is not mounted and will be mounted now."
    sudo mount /dev/sdb /data2
fi

sudo sysctl kernel.panic=15