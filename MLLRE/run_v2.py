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
    # loss.backward()
    return all_scores, loss

def evaluate_model(model, testing_data, batch_size, all_relations, device,
                   reverse_model=None):
    """
    评价模型
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
    parser.add_argument('--glove_file', default='dataset/glove.6B.50d.txt',
                        help='glove embedding file')
    parser.add_argument('--embedding_dim', default=50, type=int,
                        help='word embeddings dimensional')
    parser.add_argument('--hidden_dim', default=200, type=int,
                        help='BiLSTM hidden dimensional')
    parser.add_argument('--task_arrange', default='random',
                        help='task arrangement method, e.g. cluster_by_glove_embedding, random')
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
    parser.add_argument('--outside_epoch', default=200, type=float,
                        help='task level epoch')
    parser.add_argument('--early_stop', default=20, type=float,
                        help='task level epoch')
    parser.add_argument('--step_size', default=0.4, type=float,
                        help='step size Epsilon')
    parser.add_argument('--learning_rate', default=2e-2, type=float,
                        help='learning rate')
    parser.add_argument('--random_seed', default=317, type=int,
                        help='random seed')
    parser.add_argument('--task_memory_size', default=50, type=int,
                        help='number of samples for each task')
    parser.add_argument('--memory_select_method', default='vec_cluster',
                        help='the method of sample memory data, e.g. vec_cluster, random, difficulty')



    opt = parser.parse_args()
    print(opt)
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    np.random.RandomState(opt.random_seed)

    device = torch.device(('cuda:%d' % opt.cuda_id) if torch.cuda.is_available() and opt.cuda_id >= 0 else 'cpu')

    # do following process
    split_train_data, train_data_dict, split_test_data, test_data_dict, split_valid_data, valid_data_dict, \
    relation_numbers, rel_features, vocabulary, embedding  = \
        load_data(opt.train_file, opt.valid_file, opt.test_file, opt.relation_file, opt.glove_file,
                  opt.embedding_dim, opt.task_arrange, opt.rel_encode, opt.task_num,
                  opt.train_instance_num)
    # prepare model
    inner_model = SimilarityModel(opt.embedding_dim, opt.hidden_dim, len(vocabulary),
                                  np.array(embedding), 1, device)

    memory_data = []  # B
    seen_relations = []
    for task_index in range(opt.task_num):  # outside loop
        # reptile start model parameters pi
        weights_before = deepcopy(inner_model.state_dict())
        train_task = split_train_data[task_index]
        # test_task = split_test_data[task_index]
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

        t = tqdm(range(opt.outside_epoch))
        best_valid_acc = 0.0
        early_stop = 0
        best_checkpoint = ''
        for epoch in t:
            weights_task = deepcopy(weights_before)
            # optimizer.zero_grad()
            # inner_model.load_state_dict({name: weights_before[name] for name in weights_before})
            batch_num = (len(current_train_data) - 1) // opt.train_instance_num + 1
            total_loss = 0.0
            weights_list = [None] * (batch_num + len(memory_data))
            for batch in range(batch_num):
                # one relation's train data
                batch_train_data = current_train_data[batch * opt.train_instance_num: (batch + 1) * opt.train_instance_num]
                inner_model.load_state_dict(weights_task)
                optimizer = optim.SGD(inner_model.parameters(), lr=opt.learning_rate)
                optimizer.zero_grad()
                scores, loss = feed_samples(inner_model, batch_train_data, loss_function, relation_numbers, device)

                loss.backward()  # 计算反向传播梯度
                # 更新参数
                # for f in inner_model.parameters():
                #     f.data.sub_(f.grad.data * opt.learning_rate)
                optimizer.step()  # 更新参数
                total_loss += loss
                weights_list[batch] = deepcopy(inner_model.state_dict())  # 保存theta_t^i

            if len(memory_data) > 0:
                for i in range(len(memory_data)):
                    one_batch_memory = memory_data[i]
                    inner_model.load_state_dict(weights_task)
                    optimizer = optim.SGD(inner_model.parameters(), lr=opt.learning_rate)
                    optimizer.zero_grad()
                    scores, loss = feed_samples(inner_model, one_batch_memory, loss_function, relation_numbers, device)

                    loss.backward()
                    # 更新参数
                    # for f in inner_model.parameters():
                    #     f.data.sub_(f.grad.data * opt.learning_rate)
                    optimizer.step()
                    total_loss += loss
                    weights_list[batch_num + i] = deepcopy(inner_model.state_dict())

            outer_step_size = opt.step_size * (1 / len(weights_list))
            for name in weights_before:
                weights_task[name] = weights_before[name] - outer_step_size * sum([weights[name] - weights_before[name]
                                                                                   for weights in weights_list])

            # load state dict of weights_after
            inner_model.load_state_dict(weights_task)
            # weights_before = deepcopy(inner_model.state_dict())

            del weights_list

            valid_acc = evaluate_model(inner_model, current_valid_data, opt.batch_size, relation_numbers, device)

            checkpoint = {'net_state': inner_model.state_dict()}
            if valid_acc > best_valid_acc:
                best_checkpoint = './checkpoint/checkpoint_task%d_epoch%d.pth.tar' % (task_index, epoch)
                torch.save(checkpoint, best_checkpoint)
                best_valid_acc = valid_acc
                early_stop = 0
            else:
                early_stop += 1

            # print()
            t.set_description('Task %i Epoch %i' % (task_index+1, epoch+1))
            t.set_postfix(loss=total_loss.item(), valid_acc=valid_acc, early_stop=early_stop, best_checkpoint=best_checkpoint)
            t.update(1)

            if early_stop >= opt.early_stop:
                # 已经充分训练了
                break
        t.close()

        # sample memory from current_train_data
        if opt.memory_select_method == 'random':
            memory_data.append(random_select_data(current_train_data, opt.task_memory_size))
        elif opt.memory_select_method == 'vec_cluster':
            memory_data.append(select_data(inner_model, current_train_data, opt.task_memory_size,
                                           relation_numbers, opt.batch_size, device))  # memorydata是一个list，list中的每个元素都是一个包含selected_num个sample的list
        elif opt.memory_select_method == 'difficulty':
            memory_data.append()

        results = [evaluate_model(inner_model, test_data, opt.batch_size, relation_numbers, device)
                   for test_data in current_test_data]  # 使用current model和alignment model对test data进行一个预测

        print(results)

if __name__ == '__main__':
    main()