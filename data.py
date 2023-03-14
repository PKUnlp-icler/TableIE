import numpy as np
import ujson as json


def read_data(file, tokenizer):
    with open('./dataset/schema/entity.json') as f:
        entity_schema = json.load(f)
    with open('./dataset/schema/relation.json') as f:
        relation_schema = json.load(f)
    with open('./dataset/schema/event.json') as f:
        event_schema = json.load(f)
    with open('./dataset/schema/role.json') as f:
        role_schema = json.load(f)

    data = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    features = []
    for sample in data:
        sentence, ner, re, ee = sample['tokens'], sample['entity_mentions'], sample['relation_mentions'], sample['event_mentions']
        if len(sentence) == 1 or len(sentence) > 500:
            continue

        tokens, token_lens = [], []
        for word in sentence:
            token = tokenizer.tokenize(word)
            token_lens.append(len(tokens))
            tokens.extend(token)
        token_lens.append(len(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        n = len(tokens)
        table1, table2 = np.zeros([n, n], dtype=int), np.zeros([n, n], dtype=int)
        ner_list, re_list, ed_list, eae_list = [], [], [], []

        for entity in ner:
            num, start, end = entity['entity_type'], entity['start'], entity['end']
            start, end = token_lens[start], token_lens[end] - 1
            ner_list.append((start, end, 2 * entity_schema.index(num) + 1))
            for kk in range(start, end + 1):
                if kk == start:
                    table1[kk][kk] = 2 * entity_schema.index(num) + 1
                else:
                    table1[kk][kk] = 2 * entity_schema.index(num) + 2

        for relation in re:
            num, entities = relation['relation_type'], relation['arguments']
            sub, obj = entities[0], entities[1]
            for entity in ner:
                if entity['id'] == sub['entity_id']:
                    start1, end1 = entity['start'], entity['end']
                    start1, end1 = token_lens[start1], token_lens[end1] - 1
                if entity['id'] == obj['entity_id']:
                    start2, end2 = entity['start'], entity['end']
                    start2, end2 = token_lens[start2], token_lens[end2] - 1
            num = relation_schema.index(num) + 1
            re_list.append((start1, end1, start2, end2, num))
            for ii in range(start1, end1 + 1):
                for jj in range(start2, end2 + 1):
                    table1[ii][jj] = num

        for event in ee:
            num, trigger, arguments = event['event_type'], event['trigger'], event['arguments']

            start1, end1 = trigger['start'], trigger['end']
            start1, end1 = token_lens[start1], token_lens[end1] - 1
            ed_list.append((start1, end1, 2 * event_schema.index(num) + 1))
            for kk in range(start1, end1 + 1):
                if kk == start1:
                    table2[kk][kk] = 2 * event_schema.index(num) + 1
                else:
                    table2[kk][kk] = 2 * event_schema.index(num) + 2

            for argument in arguments:
                for entity in ner:
                    if entity['id'] == argument['entity_id']:
                        start2, end2 = entity['start'], entity['end']
                        start2, end2 = token_lens[start2], token_lens[end2] - 1
                num2 = role_schema.index(argument['role']) + 1
                eae_list.append((2 * event_schema.index(num) + 1, start2, end2, num2))
                for ii in range(start1, end1 + 1):
                    for jj in range(start2, end2 + 1):
                        table2[ii][jj] = num2
                        table2[jj][ii] = num2
        feature = {'input_ids': input_ids, 'table1': table1, 'table2': table2, 'ner': ner_list, 're': re_list, 'ed': ed_list, 'eae': eae_list}
        features.append(feature)

    return features
