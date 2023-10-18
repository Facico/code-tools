from pyserini.search.lucene import LuceneSearcher
import jsonlines
import json
import csv
from tqdm import tqdm
from transformers import BartModel
import time
import os

def search_bm25(tokenized_query, id, bm25, k=100):
    #tokenized_query = x[0]
    #i_query = x[1]
    #bm25 = x[2]
    result = {}
    hits = searcher.search(tokenized_query, k=100)
    if len(hits) < 100:
        print(tokenized_query)
    result[id] = {'candidate_doc': [hits[i].docid for i in range(len(hits))], 'score': [hits[i].score for i in range(len(hits))]}
    return result

if __name__ == "__main__":
    # create index jsonl
    steps = ['3']
    passage_tsv = '/data1/fch123/msmarco/collection.tsv'
    query_tsv = '/data1/fch123/msmarco/queries.train.tsv'
    query_position_id_path = '/data1/fch123/msmarco/qrels.train.tsv'

    dev_query_tsv = '/data1/fch123/msmarco/queries.dev.small.tsv'
    dev_query_position_id_path = '/data1/fch123/msmarco/qrels.dev.small.tsv'
    qid2query = {}
    pid2passage = {}
    q2pos = {}
    passage_index_path = '/data1/fch123/msmarco/passage_index/passage.jsonl'
    if '1' in steps:
        
        f_index = jsonlines.open(passage_index_path, 'w')
        with open(passage_tsv, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [pid, passage] in tqdm(tsvreader, ):
                #pid2passage[pid] = passage
                f_index.write({'id': pid, 'contents': passage.lower()})
    if '2' in steps:
        q_tot = 0
        query_id = []
        with open(query_position_id_path, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tqdm(tsvreader):
                topicid = int(topicid)
                docid = int(docid)
                if str(topicid) in q2pos:
                    q2pos[str(topicid)].append(docid)
                else:
                    query_id.append(str(topicid))
                    q2pos[str(topicid)] = [docid]
                    q_tot += 1
        
        with open(query_tsv, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [qid, query] in tsvreader:
                qid2query[qid] = query
        print(q_tot)
        searcher = LuceneSearcher('/data1/fch123/msmarco/passage_index/')
        bm25_result = []
        time_start=time.time()
        for i in tqdm(range(len(query_id))):
            query = qid2query[query_id[i]].lower()
            bm25_result.append(search_bm25(query, query_id[i], searcher))
        end_start=time.time()
        print('cost time {:.5f} min'.format((end_start - time_start)/ 60 ))

        result_path = '/data1/fch123/msmarco/bm25_result.json'
        with open(result_path, 'w') as f:
            json.dump(bm25_result, f, indent=2)
    
    if '3' in steps:
        trec_path = '/data1/fch123/msmarco/bm25_result.trec'
        result_path = '/data1/fch123/msmarco/bm25_result.json'
        tot = 0
        with open(trec_path, 'w') as f:
            with open(result_path, 'r') as ff:
                bm25_data = json.load(ff)
                for i in tqdm(range(len(bm25_data))):
                    qid = 0
                    for k, v in bm25_data[i].items():
                        qid = k
                        break
                    if len(bm25_data[i][qid]["candidate_doc"]) < 100:
                        continue
                    for j in range(100):
                        try:
                            docid, score = bm25_data[i][qid]["candidate_doc"][j], bm25_data[i][qid]["score"][j]
                            f.write("{} Q0 {} {} {} bm25".format(qid, docid, j+1, score))
                            f.write('\n')
                        except:
                            #print(len(bm25_data[i][qid]["candidate_doc"]))
                            #print(len(bm25_data[i][qid]["score"]))
                            if j == 0:
                                break
                            docid, score = bm25_data[i][qid]["candidate_doc"][0], bm25_data[i][qid]["score"][0]
                            f.write("{} Q0 {} {} {} bm25".format(qid, docid, j+1, score))
                            f.write('\n')
                            tot += 1
                print(tot)
                    
