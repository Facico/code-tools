## bm25
install pyserini and JAVA->
```
pip install pyserini faiss-gpu
apt search openjdk
sudo apt install openjdk-18-jdk
```

1、生成passage_jsonl文件=>生成index bm25_index.sh
2、生成query文件=>batch inference bm25_inference.sh(也可以用bm25_serini.py中的单步inference)
3、draw