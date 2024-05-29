#!/usr/bin/env python3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import jieba
import time
import csv


def similarity(cut, score):
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\w+|\.+')
    tfidf_matrix = vectorizer.fit_transform(cut)

    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(cosine_sim_matrix, index=cut.index, columns=cut.index)

    # 找出所有高于阈值的相似度对
    high_similarity_pairs = similarity_df.where(similarity_df > score).stack().dropna()

    # 获取这些相似度对的索引（物料名称对）
    similar_pairs_indices = high_similarity_pairs.index.tolist()

    # 构建分组
    groups = {}
    for pair in similar_pairs_indices:
        if not any(pair[1] in group for group in groups.values()):
            # 添加物料名称到对应的组，如果组不存在则创建新组
            for item in pair:
                if item not in groups:
                    groups[item] = {item}
                else:
                    groups[item].add(pair[1])  # 添加另一项到组中，注意这里假设pair是有序的，且我们不重复添加相同的对

    # 过滤出所有成员超过1个（即本身）的组，意味着这些物料至少与另一个物料相似
    return {key: group for key, group in groups.items() if len(group) > 1}


wl = pd.read_csv('86115.csv', low_memory=False)
wl['MATERIEL_NAME'] = wl['MATERIEL_NAME'].astype(str)
wl['SPEC_MODEL_NUMBER'] = wl['SPEC_MODEL_NUMBER'].astype(str)

print(time.strftime("%Y-%m-%d %H:%M:%S -- ", time.localtime()) + '开始对物料名称进行分词')
wl['NAME_CUT'] = wl['MATERIEL_NAME'].apply(lambda wl_name: " ".join(jieba.cut(wl_name)))
print(time.strftime("%Y-%m-%d %H:%M:%S -- ", time.localtime()) + '开始对规格型号进行分词')
wl['SPEC_CUT'] = wl['SPEC_MODEL_NUMBER'].apply(lambda spec_name: " ".join(jieba.cut(spec_name)))

wl_group_name = wl.drop_duplicates('MATERIEL_NAME')

print(time.strftime("%Y-%m-%d %H:%M:%S -- ", time.localtime()) + '开始打分')
non_singleton_groups = similarity(wl_group_name['NAME_CUT'], 0.5)

print(time.strftime("%Y-%m-%d %H:%M:%S -- ", time.localtime()) + '物料名称打分完成，规格型号开始')
non_slt_group_names = non_singleton_groups.values()

non_slt_group_specs = [{}]
for name in non_slt_group_names:
    name_csv = wl.loc[[item for item in name]]
    spec_csv = wl[wl['MATERIEL_NAME'].isin(name_csv['MATERIEL_NAME'])]
    non_singleton_groups_spec = similarity(spec_csv['SPEC_CUT'], 0.9)
    non_slt_group_specs += non_singleton_groups_spec.values()

print(time.strftime("%Y-%m-%d %H:%M:%S -- ", time.localtime()) + '打分完成，开始排序')
wl_new = wl.loc[[item for subset in non_slt_group_specs for item in subset]]

wl_new = wl_new.drop('NAME_CUT', axis=1)
wl_new = wl_new.drop('SPEC_CUT', axis=1)

print(time.strftime("%Y-%m-%d %H:%M:%S -- ", time.localtime()) + '结束')
wl_new.to_csv('res-86115.csv', index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
