import torch
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=104, type=int)

    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--dev_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)

    parser.add_argument("--entity_size", default=15, type=int)
    parser.add_argument("--relation_size", default=7, type=int)
    parser.add_argument("--event_size", default=67, type=int)
    parser.add_argument("--role_size", default=23, type=int)

    parser.add_argument("--hidden_size1", default=300, type=int)
    parser.add_argument("--hidden_size2", default=600, type=int)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--logits_dropout", default=0.4, type=float)

    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--bert_learning_rate", default=1e-5, type=float)
    parser.add_argument("--bert_weight_decay", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_epoch", default=80, type=int)
    parser.add_argument("--path_checkpoint", default="./checkpoint/model.pkl", type=str)
    parser.add_argument("--ner_checkpoint", default="./checkpoint/ner_best.pkl", type=str)
    parser.add_argument("--re_checkpoint", default="./checkpoint/re_best.pkl", type=str)
    parser.add_argument("--edi_checkpoint", default="./checkpoint/edi_best.pkl", type=str)
    parser.add_argument("--edc_checkpoint", default="./checkpoint/edc_best.pkl", type=str)
    parser.add_argument("--eaei_checkpoint", default="./checkpoint/eaei_best.pkl", type=str)
    parser.add_argument("--eaec_checkpoint", default="./checkpoint/eaec_best.pkl", type=str)
    args = parser.parse_args()

    return args


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])

    input_ids = [f["input_ids"] + [1] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    table1 = [f["table1"] for f in batch]
    table2 = [f["table2"] for f in batch]
    ner_list = [f["ner"] for f in batch]
    re_list = [f["re"] for f in batch]
    ed_list = [f["ed"] for f in batch]
    eae_list = [f["eae"] for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)

    output = (input_ids, input_mask, table1, table2, ner_list, re_list, ed_list, eae_list)
    return output


def get_pred(tables):
    table1, table2 = tables
    n = table1.shape[0]

    i = 0
    ner_list = []
    while i < n:
        if table1[i][i] % 2 == 1:
            num = table1[i][i]
            start_pos = i
            while i + 1 < n and table1[i+1][i+1] == num + 1:
                i += 1
            end_pos = i
            ner_list.append((start_pos, end_pos, num))
        i += 1

    re_list = []
    for i in range(len(ner_list)):
        for j in range(len(ner_list)):
            if i != j:
                start1, end1, _ = ner_list[i]
                start2, end2, _ = ner_list[j]

                l = []
                for ii in range(start1, end1 + 1):
                    for jj in range(start2, end2 + 1):
                        l.append(table1[ii][jj])

                if len(set(l)) == 1 and l[0] != 0:
                    re_list.append((start1, end1, start2, end2, l[0]))

    i = 0
    ed_list = []
    while i < n:
        if table2[i][i] % 2 == 1:
            num = table2[i][i]
            start_pos = i
            while i + 1 < n and table2[i+1][i+1] == num + 1:
                i += 1
            end_pos = i
            ed_list.append((start_pos, end_pos, num))
        i += 1

    eae_list = []
    for i in range(len(ed_list)):
        for j in range(len(ner_list)):
            start1, end1, num = ed_list[i]
            start2, end2, _ = ner_list[j]

            l = []
            for ii in range(start1, end1 + 1):
                for jj in range(start2, end2 + 1):
                    l.append(table2[ii][jj])

            if len(set(l)) == 1 and l[0] != 0:
                eae_list.append((num, start2, end2, l[0]))

    return ner_list, re_list, ed_list, eae_list


def f1_eval(results_all, labels_all):
    ner_correct, ner_predict, ner_label = 0, 0, 0
    re_correct, re_predict, re_label = 0, 0, 0
    edid_correct, ed_correct, ed_predict, ed_label = 0, 0, 0, 0
    eaeid_correct, eae_correct, eae_predict, eae_label = 0, 0, 0, 0

    for i in range(len(results_all)):
        result, label = results_all[i], labels_all[i]

        result_ner, label_ner = result[0], label[0]
        for res in result_ner:
            if res in label_ner:
                ner_correct += 1
        ner_predict += len(result_ner)
        ner_label += len(label_ner)

        result_re, label_re = result[1], label[1]
        for res in result_re:
            if res in label_re:
                re_correct += 1
        re_predict += len(result_re)
        re_label += len(label_re)

        result_ed, label_ed = result[2], label[2]
        for res in result_ed:
            if res in label_ed:
                ed_correct += 1
        ed_predict += len(result_ed)
        ed_label += len(label_ed)

        result_edid, label_edid = [], []
        for j in range(len(result_ed)):
            result_edid.append((result_ed[j][0], result_ed[j][1]))
        for j in range(len(label_ed)):
            label_edid.append((label_ed[j][0], label_ed[j][1]))
        for res in result_edid:
            if res in label_edid:
                edid_correct += 1

        result_eae, label_eae = result[3], label[3]
        for res in result_eae:
            if res in label_eae:
                eae_correct += 1
        eae_predict += len(result_eae)
        eae_label += len(label_eae)

        result_eaeid, label_eaeid = [], []
        for j in range(len(result_eae)):
            result_eaeid.append((result_eae[j][0], result_eae[j][1], result_eae[j][2]))
        for j in range(len(label_eae)):
            label_eaeid.append((label_eae[j][0], label_eae[j][1], label_eae[j][2]))
        for res in result_eaeid:
            if res in label_eaeid:
                eaeid_correct += 1

    f = [0.0] * 6
    p = [ner_correct / ner_predict if ner_predict != 0 else 1, re_correct / re_predict if re_predict != 0 else 1,
         edid_correct / ed_predict if ed_predict != 0 else 1, ed_correct / ed_predict if ed_predict != 0 else 1,
         eaeid_correct / eae_predict if eae_predict != 0 else 1, eae_correct / eae_predict if eae_predict != 0 else 1]
    r = [ner_correct / ner_label, re_correct / re_label, edid_correct / ed_label, ed_correct / ed_label,
         eaeid_correct / eae_label, eae_correct / eae_label]
    for i in range(6):
        f[i] = 2 * p[i] * r[i] / (p[i] + r[i]) if p[i] + r[i] != 0 else 0
    return f
