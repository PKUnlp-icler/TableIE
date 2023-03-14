import torch
import torch.nn as nn
import torch.nn.functional as F


class Table(nn.Module):
    def __init__(self, model, args):
        super(Table, self).__init__()
        self.bert = model
        self.args = args
        self.config = model.config

        self.bert_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        self.logits_dropout = nn.Dropout(args.logits_dropout)

        self.re_table0 = nn.Linear(self.config.hidden_size * 2 + self.config.num_attention_heads * self.config.num_hidden_layers, self.config.hidden_size)
        self.ee_table0 = nn.Linear(self.config.hidden_size * 2 + self.config.num_attention_heads * self.config.num_hidden_layers, self.config.hidden_size)

        self.re_table1 = nn.Linear(self.config.hidden_size * 2 + self.config.num_attention_heads * self.config.num_hidden_layers, self.config.hidden_size)
        self.ee_table1 = nn.Linear(self.config.hidden_size * 2 + self.config.num_attention_heads * self.config.num_hidden_layers, self.config.hidden_size)

        self.re_table2 = nn.Linear(self.config.hidden_size * 2 + self.config.num_attention_heads * self.config.num_hidden_layers, self.config.hidden_size)
        self.ee_table2 = nn.Linear(self.config.hidden_size * 2 + self.config.num_attention_heads * self.config.num_hidden_layers, self.config.hidden_size)

        self.norm1 = nn.LayerNorm(self.config.hidden_size)
        self.norm2 = nn.LayerNorm(self.config.hidden_size)
        self.norm3 = nn.LayerNorm(self.config.hidden_size)
        self.norm4 = nn.LayerNorm(self.config.hidden_size)

        self.re_seq1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.ee_seq1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.re_seq2 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.ee_seq2 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)

        self.re_bin1 = nn.Linear(self.config.hidden_size, args.hidden_size1)
        self.ee_bin1 = nn.Linear(self.config.hidden_size, args.hidden_size1)
        self.re_bin2 = nn.Linear(args.hidden_size1, 1)
        self.ee_bin2 = nn.Linear(args.hidden_size1, 1)

        self.re_bin3 = nn.Linear(self.config.hidden_size, args.hidden_size1)
        self.ee_bin3 = nn.Linear(self.config.hidden_size, args.hidden_size1)
        self.re_bin4 = nn.Linear(args.hidden_size1, 1)
        self.ee_bin4 = nn.Linear(args.hidden_size1, 1)

        self.entity_out1 = nn.Linear(self.config.hidden_size, args.hidden_size2)
        self.relation_out1 = nn.Linear(self.config.hidden_size, args.hidden_size2)
        self.trigger_out1 = nn.Linear(self.config.hidden_size, args.hidden_size2)
        self.argument_out1 = nn.Linear(self.config.hidden_size, args.hidden_size2)
        self.entity_out2 = nn.Linear(args.hidden_size2, args.entity_size)
        self.relation_out2 = nn.Linear(args.hidden_size2, args.relation_size)
        self.trigger_out2 = nn.Linear(args.hidden_size2, args.event_size)
        self.argument_out2 = nn.Linear(args.hidden_size2, args.role_size)

    def forward(self, input_ids, input_mask, table1, table2):
        output = self.bert(input_ids=input_ids, attention_mask=input_mask, output_attentions=True, output_hidden_states=True)
        sequence_output = self.bert_dropout(output[0])

        x, y = torch.broadcast_tensors(sequence_output[:, :, None], sequence_output[:, None])
        z = torch.stack((output[-1]), dim=0).permute(1, 3, 4, 0, 2).flatten(3, 4)
        re_table_embedding = self.dropout(F.gelu(self.re_table0(torch.cat([x, y, z], dim=-1))))
        ee_table_embedding = self.dropout(F.gelu(self.ee_table0(torch.cat([x, y, z], dim=-1))))

        total_loss = 0
        results = []
        for k in range(len(input_ids)):
            n = table1[k].shape[0]

            mask1 = torch.eye(n).int().to(input_ids.device)
            mask2 = (torch.ones(n, n) - torch.eye(n)).int().to(input_ids.device)

            labels1 = torch.tensor(table1[k]).to(input_ids.device)
            labels2 = torch.tensor(table2[k]).to(input_ids.device)

            re_table0, ee_table0 = re_table_embedding[k, 1:n + 1, 1:n + 1], ee_table_embedding[k, 1:n + 1, 1:n + 1]

            context_information = sequence_output[k, 1:n + 1]

            re_table1, ee_table1, loss_bin1, loss_bin2 = self.within_table(k, n, re_table0, ee_table0, context_information, z, labels1, labels2)
            re_table2, ee_table2, loss_bin3, loss_bin4 = self.cross_table(k, n, re_table1, ee_table1, context_information, z, labels1, labels2)

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            entity_logits = self.entity_out2(self.dropout(F.gelu(self.entity_out1(re_table2))))
            loss1 = loss_fct(self.logits_dropout(entity_logits).flatten(0, 1), (labels1 * mask1).flatten())
            entity_loss = torch.masked_select(loss1, mask1.bool().flatten()).mean()

            relation_logits = self.relation_out2(self.dropout(F.gelu(self.relation_out1(re_table2))))
            loss2 = loss_fct(self.logits_dropout(relation_logits).flatten(0, 1), (labels1 * mask2).flatten())
            relation_loss = torch.masked_select(loss2, mask2.bool().flatten()).mean()

            trigger_logits = self.trigger_out2(self.dropout(F.gelu(self.trigger_out1(ee_table2))))
            loss3 = loss_fct(self.logits_dropout(trigger_logits).flatten(0, 1), (labels2 * mask1).flatten())
            trigger_loss = torch.masked_select(loss3, mask1.bool().flatten()).mean()

            argument_logits = self.argument_out2(self.dropout(F.gelu(self.argument_out1(ee_table2))))
            loss4 = loss_fct(self.logits_dropout(argument_logits).flatten(0, 1), (labels2 * mask2).flatten())
            argument_loss = torch.masked_select(loss4, mask2.bool().flatten()).mean()

            loss = entity_loss + trigger_loss + relation_loss + argument_loss + 3 * (loss_bin1 + loss_bin2 + loss_bin3 + loss_bin4)
            total_loss += loss

            entity_res = torch.argmax(entity_logits, dim=2)
            relation_res = torch.argmax(relation_logits, dim=2)
            trigger_res = torch.argmax(trigger_logits, dim=2)
            argument_res = torch.argmax(argument_logits + argument_logits.transpose(0, 1), dim=2)
            res1 = entity_res * mask1 + relation_res * mask2
            res2 = trigger_res * mask1 + argument_res * mask2

            results.append((res1.to('cpu').numpy(), res2.to('cpu').numpy()))

        return total_loss / len(input_ids), results

    def within_table(self, k, n, re_table, ee_table, context_information, z, labels1, labels2):
        re_mask_logits = self.re_bin2(self.dropout(F.gelu(self.re_bin1(re_table))))
        ee_mask_logits = self.ee_bin2(self.dropout(F.gelu(self.ee_bin1(ee_table))))

        re_mask = torch.sigmoid(re_mask_logits)
        ee_mask = torch.sigmoid(ee_mask_logits)

        re_information = re_table * re_mask
        re_information = self.norm1(re_information.sum(0) + re_information.sum(1))

        ee_information = ee_table * ee_mask
        ee_information = self.norm2(ee_information.sum(0) + ee_information.sum(1))

        new_re = self.dropout(F.gelu(self.re_seq1(torch.cat((context_information, re_information), dim=-1))))
        new_ee = self.dropout(F.gelu(self.ee_seq1(torch.cat((context_information, ee_information), dim=-1))))

        x, y = torch.broadcast_tensors(new_re[:, None], new_re[None])
        re_table = self.dropout(F.gelu(self.re_table1(torch.cat([x, y, z[k, 1:n + 1, 1:n + 1, :]], dim=-1))))

        x, y = torch.broadcast_tensors(new_ee[:, None], new_ee[None])
        ee_table = self.dropout(F.gelu(self.ee_table1(torch.cat([x, y, z[k, 1:n + 1, 1:n + 1, :]], dim=-1))))

        loss_fct = nn.BCEWithLogitsLoss()
        loss_bin1 = loss_fct(re_mask_logits.squeeze(2).flatten(), labels1.clamp(0, 1).flatten().float())
        loss_bin2 = loss_fct(ee_mask_logits.squeeze(2).flatten(), labels2.clamp(0, 1).flatten().float())

        return re_table, ee_table, loss_bin1, loss_bin2

    def cross_table(self, k, n, re_table, ee_table, context_information, z, labels1, labels2):
        re_mask_logits = self.re_bin4(self.dropout(F.gelu(self.re_bin3(re_table))))
        ee_mask_logits = self.ee_bin4(self.dropout(F.gelu(self.ee_bin3(ee_table))))

        re_mask = torch.sigmoid(re_mask_logits)
        ee_mask = torch.sigmoid(ee_mask_logits)

        re_information = re_table * re_mask
        re_information = self.norm3(re_information.sum(0) + re_information.sum(1))

        ee_information = ee_table * ee_mask
        ee_information = self.norm4(ee_information.sum(0) + ee_information.sum(1))

        new_re = self.dropout(F.gelu(self.re_seq2(torch.cat((context_information, ee_information), dim=-1))))
        new_ee = self.dropout(F.gelu(self.ee_seq2(torch.cat((context_information, re_information), dim=-1))))

        x, y = torch.broadcast_tensors(new_re[:, None], new_re[None])
        re_table = self.dropout(F.gelu(self.re_table2(torch.cat([x, y, z[k, 1:n + 1, 1:n + 1, :]], dim=-1))))

        x, y = torch.broadcast_tensors(new_ee[:, None], new_ee[None])
        ee_table = self.dropout(F.gelu(self.ee_table2(torch.cat([x, y, z[k, 1:n + 1, 1:n + 1, :]], dim=-1))))

        loss_fct = nn.BCEWithLogitsLoss()
        loss_bin1 = loss_fct(re_mask_logits.squeeze(2).flatten(), labels1.clamp(0, 1).flatten().float())
        loss_bin2 = loss_fct(ee_mask_logits.squeeze(2).flatten(), labels2.clamp(0, 1).flatten().float())

        return re_table, ee_table, loss_bin1, loss_bin2
