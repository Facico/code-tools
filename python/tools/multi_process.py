from multiprocessing import Pool
def search_bm25(tokenized_query, id, bm25, k=100):
    #tokenized_query = x[0]
    #i_query = x[1]
    #bm25 = x[2]
    result = {}
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:k]
    score = [scores[i] for i in top_n]
    result[id] = {'candidate_doc': [corpus_name[i] for i in top_n], 'score': score}
    return result
time_start=time.time()
with Pool(cores) as p:
    _result = list((tqdm(p.starmap(search_bm25, tokenized_query), total=len(tokenized_query), desc='bm25')))
    for i in _result:
        bm25_result.append(i)
end_start=time.time()
print('cost time {:.5f} min'.format((end_start - time_start)/ 60 ))

## map
rs = pool.map(harvest_tables, filenames)
tables = []
for r in rs:
    tables = tables + r