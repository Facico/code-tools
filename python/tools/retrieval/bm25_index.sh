python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /data1/fch123/msmarco/sample_jsonl \
  --index /data1/fch123/msmarco/passage_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 2 \
  --storePositions --storeDocvectors --storeRaw
