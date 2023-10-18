python -m pyserini.search.lucene \
    --index /home/lzy/nips/data/platypus/match/bigbench_index \
    --topics /home/lzy/nips/data/platypus/match/test_query/query_$target.tsv \
    --output /home/lzy/nips/data/platypus/match/trec/run.bigbench.$target.txt \
    --bm25 \
    --max-passage-hits 1000 --threads 32 --batch-size 64