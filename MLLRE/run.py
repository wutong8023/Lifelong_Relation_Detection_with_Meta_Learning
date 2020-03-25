import sys

from tqdm import tqdm

from utils import *
from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
from model import SimilarityModel
from copy import deepcopy
import torch.optim as optim

# process the data by adding questions
def process_testing_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    gold_relation_indexs = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        gold_relation_indexs.append(sample[0])
        neg_relations = [torch.tensor(all_relations[index],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]
        relation_set_lengths.append(len(neg_relations))
        relations += neg_relations
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return gold_relation_indexs, questions, relations, relation_set_lengths

def process_samples(sample_list, all_relations, device):
    questions = []
    relations = []
    relation_set_lengths = []
    for sample in sample_list:
        question = torch.tensor(sample[2], dtype=torch.long).to(device)
        #print(relations[sample[0]])
        #print(sample)
        pos_relation = torch.tensor(all_relations[sample[0]],
                                    dtype=torch.long).to(device)  # 正确关系的tensor，一维数组
        neg_relations = [torch.tensor(all_relations[index],
                                      dtype=torch.long).to(device)
                         for index in sample[1]]  # 候选的错误关系tensor
        relation_set_lengths.append(len(neg_relations)+1)
        relations += [pos_relation] + neg_relations  # 合并
        #questions += [question for i in range(relation_set_lengths[-1])]
        questions += [question] * relation_set_lengths[-1]
    return questions, relations, relation_set_lengths

def ranking_sequence(sequence):
    word_lengths = torch.tensor([len(sentence) for sentence in sequence])
    rankedi_word, indexs = word_lengths.sort(descending = True)
    ranked_indexs, inverse_indexs = indexs.sort()
    #print(indexs)
    sequence = [sequence[i] for i in indexs]
    return sequence, inverse_indexs

def feed_samples(model, samples, loss_function, all_relations, device,
                 alignment_model=None):
    """

    :param model: SimilarityModel
    :param samples: 一个batch的训练数据
    :param loss_function: MarginLoss：计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0
    :param all_relations: 全部关系包括/fill/fill/fill的word list的list。 = [[rel_0_word_indices], [rel_1_word_indices], ..., [rel_80_word_indices]]
    :param device:
    :param alignment_model:
    :return:
    """
    questions, relations, relation_set_lengths = process_samples(
        samples, all_relations, device)  # 将每个sample都进行展开，做成question和一个候选关系一对一的形式，relation_set_lengths记录了每个sample展开成了几个句子
    ranked_questions, alignment_question_indexs = \
        ranking_sequence(questions)  # 输入一个一维tensor的list，对其中的每一个list，按其中元素的长度进行排序，从大到小排序，返回排序后的list和对应原序列中的index
    ranked_relations, alignment_relation_indexs = \
        ranking_sequence(relations)
    question_lengths = [len(question) for question in ranked_questions]  # 排序之后每个question list中句子的长度
    relation_lengths = [len(relation) for relation in ranked_relations]  # 排序之后每个relation list中句子的长度
    pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)  # 进行补齐
    pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
    pad_questions = pad_questions.to(device)
    pad_relations = pad_relations.to(device)

    model.zero_grad()
    if alignment_model is not None:
        alignment_model.zero_grad()
    model.init_hidden(device, sum(relation_set_lengths))
    all_scores = model(pad_questions, pad_relations, device,
                       alignment_question_indexs, alignment_relation_indexs,
                       question_lengths, relation_lengths, alignment_model)  # 每个句子和关系对的similarity score
    all_scores = all_scores.to('cpu')
    pos_scores = []
    neg_scores = []
    pos_index = []
    start_index = 0
    for length in relation_set_lengths:
        pos_index.append(start_index)
        pos_scores.append(all_scores[start_index].expand(length-1))
        neg_scores.append(all_scores[start_index+1:start_index+length])
        start_index += length
    pos_scores = torch.cat(pos_scores)
    neg_scores = torch.cat(neg_scores)
    alignment_model_criterion = nn.MSELoss()

    loss = loss_function(pos_scores, neg_scores,
                         torch.ones(sum(relation_set_lengths)-
                                    len(relation_set_lengths)))
    loss.backward()
    return all_scores, loss

def evaluate_model(model, testing_data, batch_size, all_relations, device,
                   reverse_model=None):
    """

    :param model:
    :param testing_data:
    :param batch_size:
    :param all_relations:
    :param device:
    :param reverse_model:
    :return:
    """
    #print('start evaluate')
    num_correct = 0
    #testing_data = testing_data[0:100]
    for i in range((len(testing_data)-1)//batch_size+1):
        samples = testing_data[i*batch_size:(i+1)*batch_size]
        gold_relation_indexs, questions, relations, relation_set_lengths = \
            process_testing_samples(samples, all_relations, device)
        model.init_hidden(device, sum(relation_set_lengths))
        ranked_questions, reverse_question_indexs = \
            ranking_sequence(questions)
        ranked_relations, reverse_relation_indexs = \
            ranking_sequence(relations)
        question_lengths = [len(question) for question in ranked_questions]
        relation_lengths = [len(relation) for relation in ranked_relations]
        #print(ranked_questions)
        pad_questions = torch.nn.utils.rnn.pad_sequence(ranked_questions)
        pad_relations = torch.nn.utils.rnn.pad_sequence(ranked_relations)
        all_scores = model(pad_questions, pad_relations, device,
                           reverse_question_indexs, reverse_relation_indexs,
                           question_lengths, relation_lengths, reverse_model)
        start_index = 0
        pred_indexs = []
        #print('len of relation_set:', len(relation_set_lengths))
        for j in range(len(relation_set_lengths)):
            length = relation_set_lengths[j]
            cand_indexs = samples[j][1]
            pred_index = (cand_indexs[
                all_scores[start_index:start_index+length].argmax()])
            if pred_index == gold_relation_indexs[j]:
                num_correct += 1
            #print('scores:', all_scores[start_index:start_index+length])
            #print('cand indexs:', cand_indexs)
            #print('pred, true:',pred_index, gold_relation_indexs[j])
            start_index += length
    #print(cand_scores[-1])
    #print('num correct:', num_correct)
    #print('correct rate:', float(num_correct)/len(testing_data))
    return float(num_correct)/len(testing_data)

def print_list(result):
    for num in result:
        sys.stdout.write('%.3f, ' %num)
    print('')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', default=0, type=int,
                        help='cuda device index, -1 means use cpu')
    parser.add_argument('--train_file', default='dataset/training_data.txt',
                        help='train file')
    parser.add_argument('--valid_file', default='dataset/val_data.txt',
                        help='valid file')
    parser.add_argument('--test_file', default='dataset/val_data.txt',
                        help='test file')
    parser.add_argument('--relation_file', default='dataset/relation_name.txt',
                        help='relation name file')
    parser.add_argument('--glove_file', default='dataset/glove.6B.300d.txt',
                        help='glove embedding file')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='word embeddings dimensional')
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='BiLSTM hidden dimensional')
    parser.add_argument('--task_arrange', default='random',
                        help='task arrangement method')
    parser.add_argument('--rel_encode', default='glove',
                        help='relation encode method')
    parser.add_argument('--meta_method', default='reptile',
                        help='meta learning method, maml and reptile can be choose')
    parser.add_argument('--batch_size', default=50, type=float,
                        help='Reptile inner loop batch size')
    parser.add_argument('--task_num', default=10, type=int,
                        help='number of tasks')
    parser.add_argument('--train_instance_num', default=200, type=int,
                        help='number of instances for one relation, -1 means all.')
    parser.add_argument('--loss_margin', default=0.5, type=float,
                        help='loss margin setting')
    parser.add_argument('--outside_epoch', default=40, type=float,
                        help='task level epoch')
    parser.add_argument('--step_size', default=0.5, type=float,
                        help='step size Epsilon')
    parser.add_argument('--learning_rate', default=5e-3, type=float,
                        help='learning rate')
    parser.add_argument('--num_samplers', default=50, type=int,
                        help='number of samplers selected in one task')
    parser.add_argument('--random_seed', default=317, type=int,
                        help='random seed')
    parser.add_argument('--task_memory_size', default=50, type=int,
                        help='number of samples for each task')
    parser.add_argument('--memory_select_method', default='random',
                        help='the method of sample memory data')



    opt = parser.parse_args()

    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    np.random.RandomState(opt.random_seed)

    device = torch.device(('cuda:%d' % opt.cuda_id) if torch.cuda.is_available() and opt.cuda_id >= 0 else 'cpu')

    # do following process
    split_train_data, split_test_data, split_valid_data, relation_numbers, rel_features, vocabulary, embedding = \
        load_data(opt.train_file, opt.valid_file, opt.test_file, opt.relation_file, opt.glove_file,
                            opt.embedding_dim, opt.task_arrange, opt.rel_encode, opt.task_num,
                            opt.train_instance_num)
    # prepare model
    inner_model = SimilarityModel(opt.embedding_dim, opt.hidden_dim, len(vocabulary),
                            np.array(embedding), 1, device)

    memory_data = []
    memory_question_embed = []
    memory_relation_embed = []
    sequence_results = []
    result_whole_test = []
    seen_relations = []
    all_seen_relations = []
    memory_index = 0
    for task_index in range(opt.task_num):  # outside loop
        # reptile start model parameters pi
        weights_before = deepcopy(inner_model.state_dict())

        train_task = split_train_data[task_index]
        test_task = split_test_data[task_index]
        valid_task = split_valid_data[task_index]

        # collect seen relations
        for data_item in train_task:
            if data_item[0] not in seen_relations:
                seen_relations.append(data_item[0])

        # remove unseen relations
        current_train_data = remove_unseen_relation(train_task, seen_relations)
        current_valid_data = remove_unseen_relation(valid_task, seen_relations)
        current_test_data = []
        for previous_task_id in range(task_index + 1):
            current_test_data.append(remove_unseen_relation(split_test_data[previous_task_id], seen_relations))

        # train inner_model
        loss_function = nn.MarginRankingLoss(opt.loss_margin)
        inner_model = inner_model.to(device)
        optimizer = optim.Adam(inner_model.parameters(), lr=opt.learning_rate)
        t = tqdm(range(opt.outside_epoch))
        for epoch in t:
            batch_num = (len(current_train_data) - 1) // opt.batch_size + 1
            total_loss = 0.0
            for batch in range(batch_num):
                batch_train_data = current_train_data[batch * opt.batch_size: (batch + 1) * opt.batch_size]

                if len(memory_data) > 0:
                    all_seen_data = []
                    for one_batch_memory in memory_data:
                        all_seen_data += one_batch_memory

                    memory_batch = memory_data[memory_index]

                    scores, loss = feed_samples(inner_model, memory_batch, loss_function, relation_numbers, device)

                scores, loss = feed_samples(inner_model, batch_train_data, loss_function, relation_numbers, device)
                total_loss += loss
                optimizer.step()
            # print()
            t.set_description('Task %i Epoch %i' % (task_index+1, epoch+1))
            t.set_postfix(loss=total_loss.item())
            t.update(1)
            # for param in inner_model.parameters():
            #     param.data -= opt.learning_rate * param.grad.data  # 根据梯度信息，手动step更新梯度

        weights_after = inner_model.state_dict()  # 经过inner_epoch轮次的梯度更新后weights
        outerstepsize = opt.step_size * (1 - task_index / opt.task_num)  # linear schedule
        inner_model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                               for name in weights_before})

        results = [evaluate_model(inner_model, test_data, opt.batch_size, relation_numbers, device)
                   for test_data in current_test_data]  # 使用current model和alignment model对test data进行一个预测

        # sample memory from current_train_data
        memory_data.append(select_data(inner_model, current_train_data, opt.task_memory_size,
                                       relation_numbers, opt.batch_size, device))  # memorydata是一个list，list中的每个元素都是一个包含selected_num个sample的list

        print_list(results)






    # if opt.meta_method == 'reptile':
    #     # use reptile to train model
    #
    # elif opt.meta_method == 'maml':
    #     # use reptile to train model, wait implement
    #     pass
    # else:
    #     raise Exception('meta method %s not implement' % opt.meta_method)



if __name__ == '__main__':
    main()