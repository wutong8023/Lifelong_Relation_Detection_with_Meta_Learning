import os
import random

from utils import *
import numpy as np
import wordninja
import re

def remove_return_sym(string):
    return string.rstrip('\n')

def remove_invalid_token(token_list):
    invalid_chars = ['\xa0', '\n', ' ', '\u3000', '\u2005']
    for invalid_char in invalid_chars:
        token_list = [char for char in token_list if invalid_char not in char]
    return token_list

def read_data(file_path):
    if not os.path.exists(file_path):
        raise Exception('No such file %s' % file_path)
    tmp_dir = os.path.join(os.path.dirname(file_path), 'tmp')

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_file_path = os.path.join(tmp_dir, '%s.pkl' % file_path.split('/')[-1].split('.')[0])

    if os.path.exists(tmp_file_path):
        return read_pickle(tmp_file_path)
    else:
        with open(file_path, 'r', encoding='utf8') as f:
            data_dict = {}
            data_list = []
            for line in f:
                items = line.split('\t')
                rel_idx = int(items[0])
                candidate_rel_idx = [int(idx) for idx in items[1].split()]
                tokens = remove_invalid_token(remove_return_sym(items[2]).split())
                data_list.append([rel_idx, candidate_rel_idx, tokens])
                if rel_idx not in data_dict:
                    data_dict[rel_idx] = [[rel_idx, candidate_rel_idx, tokens]]
                else:
                    data_dict[rel_idx].append([rel_idx, candidate_rel_idx, tokens])

            dump_pickle(tmp_file_path, (data_list, data_dict))
            return data_list, data_dict

def read_relation(file_path):
    if not os.path.exists(file_path):
        raise Exception('No such file %s' % file_path)

    with open(file_path, 'r', encoding='utf8') as f:
        relation_list = []
        relation_list.append('/fill/fill/fill')
        index = 1
        relation_dict = {}
        relation_dict['/fill/fill/fill'] = 0
        for line in f:
            relation_name = remove_return_sym(line)
            relation_list.append(relation_name)
            relation_dict[relation_name] = index
            index += 1

        return relation_list, relation_dict

def read_glove(glove_file):
    if not os.path.exists(glove_file):
        raise Exception('No such file %s' % glove_file)
    tmp_dir = os.path.join(os.path.dirname(glove_file), 'tmp')

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_file_path = os.path.join(tmp_dir, '%s.pkl' % glove_file.split('/')[-1].split('.')[0])

    if os.path.exists(tmp_file_path):
        glove_vocabulary, glove_embedding = read_pickle(tmp_file_path)
        return glove_vocabulary, glove_embedding

    glove_vocabulary = []
    glove_embedding = {}
    with open(glove_file, 'r', encoding='utf8') as file_in:
        for line in file_in:
            items = line.split()
            word = items[0]
            glove_vocabulary.append(word)
            glove_embedding[word] = np.asarray(items[1:], dtype='float32')
    dump_pickle(tmp_file_path, (glove_vocabulary, glove_embedding))

    return glove_vocabulary, glove_embedding

def concat_words(words):
    if len(words) > 0:
        return_str = words[0]
        for word in words[1:]:
            return_str += '_' + word
        return return_str
    else:
        return ''

def split_relation_into_words(relation, glove_vocabulary):
    word_list = []
    relation_list = []
    # some relation will have fours parts, where the first part looks like
    # "base". We only choose the last three parts
    for word_seq in relation.split("/")[-3:]:
        new_word_list = []
        #for word in word_seq.split("_"):
        for word in re.findall(r"[\w']+", word_seq):
            #print(word)
            if word not in glove_vocabulary:
                new_word_list += wordninja.split(word)
            else:
                new_word_list += [word]
        #print(new_word_list)
        word_list += new_word_list
        relation_list.append(concat_words(new_word_list))
    return word_list+relation_list

def clean_relations(relation_list, glove_vocabulary):
    cleaned_relations = []
    for relation in relation_list:
        cleaned_relations.append(split_relation_into_words(relation, glove_vocabulary))
    return cleaned_relations

def build_vocabulary_embedding(relation_list, all_samples, glove_embedding,
                               embedding_size):
    vocabulary = {}
    embedding = []
    index = 0
    np.random.seed(100)
    for relation in relation_list:
        for word in relation:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in glove vocabulary randomly
                if word in glove_embedding:
                    embedding.append(glove_embedding[word])
                else:
                    embedding.append(np.random.rand(embedding_size))
    for sample in all_samples:
        question = sample[2]
        for word in question:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
                # init the word that are not in glove vocabulary randomly
                if word in glove_embedding:
                    embedding.append(glove_embedding[word])
                else:
                    #print(word)
                    embedding.append(np.random.rand(embedding_size))

    return vocabulary, embedding

def words2indexs(word_list, vocabulary):
    index_list = []
    for word in word_list:
        index_list.append(vocabulary[word])
    return index_list

def transform_relations(relation_list, vocabulary):
    relation_ixs = []
    for relation in relation_list:
        relation_ixs.append(words2indexs(relation, vocabulary))
    return relation_ixs

# transform the words in the questions into index of the vocabulary
def transform_questions(sample_list, vocabulary):
    for sample in sample_list:
        sample[2] = words2indexs(sample[2], vocabulary)
    return sample_list

def generate_data(train_file, valid_file, test_file, relation_file, glove_file, embedding_size=300):
    train_data_list, train_data_dict = read_data(train_file)
    valid_data_list, valid_data_dict = read_data(valid_file)
    test_data_list, test_data_dict = read_data(test_file)
    relation_list, relation_dict = read_relation(relation_file)
    glove_vocabulary, glove_embedding = read_glove(glove_file)
    all_samples = train_data_list + valid_data_list + test_data_list

    cleaned_relations = clean_relations(relation_list, glove_vocabulary)

    vocabulary, embedding = build_vocabulary_embedding(cleaned_relations,
                                                       all_samples,
                                                       glove_embedding,
                                                       embedding_size)

    relation_numbers = transform_relations(cleaned_relations, vocabulary)

    return train_data_list, train_data_dict, test_data_list, test_data_dict, valid_data_list, valid_data_dict, \
           relation_numbers, vocabulary, embedding

def random_split_relation(task_num, relation_dict):
    relation_num = len(relation_dict)
    relation_idx = [index + 1 for index in range(relation_num)]
    random.shuffle(relation_idx)
    rel2label = {}
    for rel_idx in relation_idx:
        rel2label[rel_idx] = relation_idx.index(rel_idx) % task_num

    return rel2label

def split_data(data, vocabulary, rel2cluster, task_num, instance_num=-1):  # -1 means all
    separated_data = [None] * task_num
    for rel, items in data.items():
        rel_culter = rel2cluster[rel]
        if instance_num > 0 :
            selected_samples = random.sample(items, instance_num)
        else:
            selected_samples = items[:]

        if separated_data[rel_culter] is None:
            separated_data[rel_culter] = selected_samples
        else:
            separated_data[rel_culter].extend(selected_samples)

    for i in range(len(separated_data)):
        separated_data[i] = transform_questions(separated_data[i], vocabulary)
    return separated_data

def split_relation(relation):
    word_list = []
    # some relation will have fours parts, where the first part looks like
    # "base". We only choose the last three parts
    for word_seq in relation.split("/")[-3:]:
        for word in word_seq.split("_"):
            word_list += wordninja.split(word)
    return word_list

def get_embedding(relation_name, glove_embeddings):
    word_list = split_relation(relation_name)
    relation_embeddings = []
    for word in word_list:
        if word.lower() in glove_embeddings:
            relation_embeddings.append(glove_embeddings[word.lower()])
        else:
            print(word,"is not contained in glove vocabulary")
    return np.mean(relation_embeddings, 0)

def rel_glove_feature(relation_file, glove_file):
    relation_list, relation_dict = read_relation(relation_file)
    glove_vocabulary, glove_embedding = read_glove(glove_file)

    rel_glove_embedding = [None] * len(relation_dict)

    for rel, idx in relation_dict.items():
        rel_embedding = get_embedding(rel, glove_embedding)
        rel_glove_embedding[idx - 1] = rel_embedding  # rel_glove_embedding index from 0, relation_dict index from 1

    return rel_glove_embedding

def rel_kg_feature():
    pass

def cluster_data_by_glove(task_num, ):
    # waits implement
    pass

def cluster_data_by_kg(task_num):
    # waits implement
    pass

def load_data(train_file, valid_file, test_file, relation_file, glove_file, embedding_size=300,
              task_arrange='random', rel_encode='glove', task_num=10, instance_num=100):
    # generate data
    train_data_list, train_data_dict, test_data_list, test_data_dict, valid_data_list, valid_data_dict, \
    relation_numbers, vocabulary, embedding = \
        generate_data(train_file, valid_file, test_file, relation_file, glove_file, embedding_size)
    relation_list, relation_dict = read_relation(relation_file)

    # arrange task
    if task_arrange == 'random':
        rel2cluster = random_split_relation(task_num, relation_dict)
        if rel_encode == 'glove':
            rel_features = rel_glove_feature(relation_file, glove_file)
        elif rel_encode == 'kg':
            rel_features = rel_kg_feature()
        else:
            raise Exception('rel_encode method %s not implement.' % rel_encode)

    elif task_arrange == 'cluster_by_glove_embedding':
        rel2cluster, rel_features = cluster_data_by_glove(task_num)
    elif task_arrange == 'cluster_by_kg_embedding':
        rel2cluster, rel_features = cluster_data_by_kg(task_num)
    else:
        raise Exception('task arrangement method %s not implement' % task_arrange)

    split_train_data = split_data(train_data_dict, vocabulary, rel2cluster, task_num, instance_num)
    split_test_data = split_data(test_data_dict, vocabulary, rel2cluster, task_num)
    split_valid_data = split_data(valid_data_dict, vocabulary, rel2cluster, task_num)

    return split_train_data, split_test_data, split_valid_data, relation_numbers, rel_features, vocabulary, embedding
