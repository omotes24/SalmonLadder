"""AdaNeg (OpenOOD-VLM TTAPromptPostprocessor_noadagap) with the sample-adaptive term and the final similarity
evaluated in chunks of test images so that one batch of 256 fits an 11 GB GPU. Everything else is the official
method body verbatim (memory update over the whole batch first, then scoring against the updated memory)."""
from typing import Any

import pdb

import torch
import torch.nn as nn

from openood.postprocessors.ttaprompt_postprocessor import TTAPromptPostprocessor_noadagap

CHUNK = 32


class AdaNegChunked(TTAPromptPostprocessor_noadagap):
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        class_num = net.n_cls
        image_features, text_features, logit_scale = net(data, return_feat=True)
        if self.reset:
            ## reset feat memory to vanilla text features, for each ID/OOD pair.
            self.feat_memory = text_features.unsqueeze(1) ## C*1*D
            extented_empty_memory = torch.zeros_like(self.feat_memory).repeat(1, self.memory_size, 1)
            self.feat_memory = torch.cat((self.feat_memory, extented_empty_memory), dim=1)
            self.indice_memory = torch.ones(text_features.size(0))
            self.entropy_memory = torch.zeros(self.feat_memory.size(0), self.feat_memory.size(1)).to(text_features.device) ## C*(1+memory_size)
            self.reset=False
        ## image_features: 256*512
        ## text_features: 11k*7*512
        ## extract sample adaptative classifier via attention
        output_vanilla = logit_scale * image_features @ text_features.t()  # batch * class, to decide assign image to which category and the corresponding confidence.
        prob_vanilla = torch.softmax(output_vanilla, dim=1)
        # conf_in_vanilla = torch.sum(prob_vanilla[:, :class_num], dim=1)
        pos_logit = output_vanilla[:, :class_num] ## B*C
        neg_logit = output_vanilla[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output_vanilla.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in_vanilla = scores.mean(dim=-1) ### the mean ID score of multiple groups. 

        # pdb.set_trace()
        # threshold = self.thres
        activate_indicator = conf_in_vanilla > (self.thres + self.gap * (1-self.thres)) ## only store high confident samples into feature memory.

        # activate_indicator = torch.randint(0, 2, (activate_indicator.shape[0],), dtype=torch.bool)  ## for rebuttal analyses with high error of pseudo labels.
        _, pred_all = torch.max(output_vanilla[:, :class_num], dim=1)
        prob_id = torch.softmax(output_vanilla[:, :class_num], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item()
                predicted_prob = prob_id[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass
        
        # activate_indicator = ~activate_indicator
        activate_indicator = conf_in_vanilla < (self.thres - self.gap * self.thres) ## only store high confident samples into feature memory.
        # pdb.set_trace()
        ####################################### in the above formulation, the default middle point is 0.5; thres - 0.5 is the unsed gap. 
        _, pred_all = torch.max(output_vanilla[:, class_num:], dim=1)
        prob_ood = torch.softmax(output_vanilla[:, class_num:], dim=1)
        for i in range(activate_indicator.size(0)):
            if activate_indicator[i].item():
                predicted_cate = pred_all[i].item() + class_num
                predicted_prob = prob_ood[i]
                current_instance_entropy = -(predicted_prob * (torch.log(predicted_prob + 1e-8))).sum()
                # pdb.set_trace()
                # self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                if self.indice_memory[predicted_cate] == self.memory_size:
                    if (current_instance_entropy < self.entropy_memory[predicted_cate]).sum() == 0:
                        pass  ## the entropy of current test image is very large.
                    else:
                        # replace the one with the maximum entropy!! to update. find the one with the maximum entropy.
                        _, indice = torch.sort(self.entropy_memory[predicted_cate])
                        to_replace_indice = indice[-1]  ## with max entropy, ascending.
                        self.feat_memory[predicted_cate][to_replace_indice] = image_features[i]
                        self.entropy_memory[predicted_cate][to_replace_indice] = current_instance_entropy
                else:
                    self.feat_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = image_features[i]
                    self.entropy_memory[predicted_cate][self.indice_memory[predicted_cate].long()] = current_instance_entropy
                    self.indice_memory[predicted_cate] += 1
            else:
                pass

        ###predicting final ood confident and ID prediction with self.feat_memory
        # --- memory-bounded evaluation of the official expressions (chunks of test images; per-image arithmetic unchanged)
        outs = []
        for b0 in range(0, image_features.size(0), CHUNK):
            imf = image_features[b0:b0 + CHUNK]
            sim = self.feat_memory @ imf.t()  ## 11k*30*b
            sim = torch.exp(-self.beta * (-sim + 1))
            if self.samada:
                sa_text_features_list = []
                split_num = int(self.feat_memory.size(0) / 1000.0)
                for i in range(split_num):
                    temp = sim[1000*i:1000*(i+1)].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*i:1000*(i+1)].unsqueeze(0)
                    sa_text_features_split = temp.sum(2)
                    sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)
                    sa_text_features_list.append(sa_text_features_split)
                if self.feat_memory.size(0) % 1000.0 > 0:
                    temp = sim[1000*split_num:].unsqueeze(0).transpose(0,-1) * self.feat_memory[1000*split_num:].unsqueeze(0)
                    sa_text_features_split = temp.sum(2)
                    sa_text_features_split /= sa_text_features_split.norm(dim=-1, keepdim=True)
                    sa_text_features_list.append(sa_text_features_split)
                sa_text_features = torch.cat(sa_text_features_list, dim=1)
            else:
                sa_text_features = self.feat_memory.mean(1)
                sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)
            outs.append(logit_scale * (imf.unsqueeze(1) * sa_text_features).sum(-1))
        output = torch.cat(outs, dim=0)
        # pdb.set_trace()
        prob_tta = torch.softmax(output, dim=1)
        prob_all = prob_vanilla + prob_tta * self.lambda_val
        # # pdb.set_trace()  ##(text_features * self.image_classifier).sum(1), around 0.3, indicating that there is a large discrepancy between text and image feat, thus they should be complementary.
        _, pred_in = torch.max(prob_all[:, :class_num], dim=1)

        pos_logit = output[:, :class_num] ## B*C
        neg_logit = output[:, class_num:] ## B*total_neg_num
        drop = neg_logit.size(1) % self.group_num
        if drop > 0:
            neg_logit = neg_logit[:, :-drop]

        if self.random_permute:
            # print('use random permute')
            SEED=0
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            idx = torch.randperm(neg_logit.shape[1]).to(output.device)
            neg_logit = neg_logit.T ## total_neg_num*B
            neg_logit = neg_logit[idx].T.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        else:
            neg_logit = neg_logit.reshape(pos_logit.shape[0], self.group_num, -1).contiguous()
        scores = []
        for i in range(self.group_num):
            full_sim = torch.cat([pos_logit, neg_logit[:, i, :]], dim=-1) 
            full_sim = full_sim.softmax(dim=-1)
            pos_score = full_sim[:, :pos_logit.shape[1]].sum(dim=-1)
            scores.append(pos_score.unsqueeze(-1))
        scores = torch.cat(scores, dim=-1)
        conf_in = scores.mean(dim=-1)
        # # ###########
        # score = torch.softmax(output, dim=1)
        # conf_in = torch.sum(score[:, :class_num], dim=1)
        # conf_out = torch.sum(score[:, class_num:], dim=1)

        # max in prob - max out prob
        if self.in_score == 'adaonly':
            conf = conf_in  ## = 1-conf_out
        elif self.in_score == 'vanillaonly':
            conf = conf_in_vanilla
        elif self.in_score == 'combine':
            conf = conf_in + conf_in_vanilla * self.lambda_val
        elif self.in_score == 'multiply':
            conf = conf_in * conf_in_vanilla
        else:
            raise NotImplementedError
        if torch.isnan(conf).any():
            pdb.set_trace()

        return pred_in, conf

