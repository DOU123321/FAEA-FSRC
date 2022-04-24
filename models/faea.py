import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class FAEA(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size, max_len):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.rel_glo_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.temp_proto = 1  # moco 0.07
        self.dropout=nn.Dropout(0.1)
        
        self.a = torch.from_numpy(np.diag(np.ones(max_len - 1, dtype=np.int32),1)).cuda()
        #对角线上移一位
        self.b = torch.from_numpy(np.diag(np.ones(max_len, dtype=np.int32),0)).cuda()
        #对角线
        self.c = torch.from_numpy(np.diag(np.ones(max_len - 1, dtype=np.int32),-1)).cuda()
        #对角线下移一位
        self.tri_matrix = torch.from_numpy(np.triu(np.ones([max_len,max_len], dtype=np.float32),0)).cuda()

        self.weight_word = nn.Parameter(torch.Tensor(self.hidden_size, 1))  
        nn.init.uniform_(self.weight_word , -0.1, 0.1)         
        self.eps=0.3

        self.current_device=True        
    def __dist__(self, x, y, dim):
        return (x * y).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def compute_constituent(self,input ,mask ):
        
        score = input.masked_fill(mask == 0, -1e9)#[10,128,128]
        
        neibor_attn1 = F.softmax(score, dim=-1) #[10,128,128]
        
        neibor_attn = torch.sqrt(neibor_attn1*neibor_attn1.transpose(-2,-1) + 1e-9)
        
        t = torch.log(neibor_attn + 1e-9).masked_fill(self.a==0, 0).matmul(self.tri_matrix)
        #[10,128,128]
        g_attn = self.tri_matrix.matmul(t).exp().masked_fill((self.tri_matrix.int()-self.b)==0, 0)     #只保留上三角
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(self.b==0, 1e-9)
        
        return g_attn 

    def forward(self, support, query, rel_text, N, K, total_Q, is_eval=False):
        """
        :param support: Inputs of the support set. (B*N*K)
        :param query: Inputs of the query set. (B*total_Q)
        :param rel_text: Inputs of the relation description.  (B*N)
        :param N: Num of classes
        :param K: Num of instances for each class in the support set
        :param total_Q: Num of instances in the query set
        :param is_eval:
        :return: logits, pred, logits_proto, labels_proto, sim_scalar
        """
        support_glo, support_loc = self.sentence_encoder(support)  # (B * N * K, 2D), (B * N * K, L, D)
        #[40, 1536](头尾实体连接)   [40, 128, 768]（每个词向量）
        query_glo, query_loc = self.sentence_encoder(query)  # (B * total_Q, 2D), (B * total_Q, L, D)
        
        rel_text_glo, rel_text_loc = self.sentence_encoder(rel_text, cat=False)  # (B * N, D), (B * N, L, D)

        # global features
        ####################################################################
        support_glo = support_glo.view(-1, N, K, self.hidden_size * 2)  # (B, N, K, 2D)
        #[4, 10, 1, 1536]
        query_glo = query_glo.view(-1, total_Q, self.hidden_size * 2)  # (B, total_Q, 2D)
        #[4,10,1536]
        rel_text_glo = self.rel_glo_linear(rel_text_glo.view(-1, N, self.hidden_size))  # (B, N, 2D)
        #[40,768]==>[4,10,1536]
        B = support_glo.shape[0] # 4
        # global prototypes  (需要,一个全局的特征)
        proto_glo = torch.mean(support_glo, 2) + rel_text_glo  # Calculate prototype for each class (B, N, 2D)
        #[4, 10, 1536]
        
        # local features
        ############################################################
        rel_text_loc_s = rel_text_loc.unsqueeze(1).expand(-1, K, -1, -1).contiguous().view(B * N * K, -1, self.hidden_size)  # (B * N * K, L, D)
        # [40, 128, 768]
       # rel_text_loc_s=torch.div(rel_text_loc_s,torch.sqrt(self.hidden_size))
        rel_support = torch.bmm(support_loc, torch.transpose(rel_text_loc_s, 2, 1))  # (B * N * K, L, L)
        #[40,128,128]
        
        #对support中每个词进行一个重要性度量，得到loc句子表示（需要改进）
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        sup_general = torch.bmm(support_loc, self.weight_word.unsqueeze(0).repeat(support_loc.size(0),1,1))
        rel_support1= rel_support  + torch.transpose(sup_general,2,1)     


        ins_att_score_s, _ = rel_support1.max(-1)  # (B * N * K, L) #找与r最相关的词  [40,128]
        ins_att_score_s=torch.div(ins_att_score_s,math.sqrt(self.hidden_size))
        support_support = torch.bmm(support_loc, torch.transpose(support_loc, 2, 1))#[10,128,128]
        support_score1 = torch.div(support_support, math.sqrt(self.hidden_size))#[10,128,128] 
        #[B,1,128]
        support_mask=support['mask'].unsqueeze(1)&(self.a+self.c)#[10,128,128] 
        
        ins_att_score_s =self.phase_attn(support_score1,support_mask,ins_att_score_s,support['mask'],support_loc)
        
        ins_att_score_s = ins_att_score_s.unsqueeze(-1)  # (B * N * K, L, 1)
        #[ 40, 128, 1]
        support_loc = torch.sum(ins_att_score_s * support_loc, dim=1)  # (B * N * K, D)
        #[40,768]
        support_loc = support_loc.view(B, N, K, self.hidden_size)
        #[4,10,1,768]
        
        #对关系描述的每个词进行重要性度量，得到loc关系描述表示（可以利用，需要改进）
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        rel_general = torch.bmm(rel_text_loc_s, self.weight_word.unsqueeze(0).repeat(rel_text_loc_s.size(0),1,1))
        rel_support2= torch.transpose(rel_support,1,2)  + torch.transpose(rel_general,2,1)     
        
        ins_att_score_r, _ = rel_support2.max(-1)  # (B * N * K, L)
        #[40,128]
        ins_att_score_r=torch.div(ins_att_score_r,math.sqrt(self.hidden_size))
        ins_att_score_r = F.softmax(torch.tanh(ins_att_score_r), dim=1).unsqueeze(-1)  # (B * N * K, L, 1)
        #[40,128,1]
        ins_att_score_r=self.dropout(ins_att_score_r)
        rel_text_loc = torch.sum(ins_att_score_r * rel_text_loc_s, dim=1).view(B, N, K, self.hidden_size)
        #[4, 10, 1, 768]
        rel_text_loc = torch.mean(rel_text_loc, 2)  # (B, N, D)
        #[4, 10,  768]
        
        #查询的自注意力，得到查询样本的loc表示（可用）
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        query_query = torch.bmm(query_loc, torch.transpose(query_loc, 2, 1))  # (B * total_Q, L, L)
        #[40,128,128]
        query_general = torch.bmm(query_loc, self.weight_word.unsqueeze(0).repeat(query_loc.size(0),1,1))
        query_query1= query_query  + torch.transpose(query_general,2,1)        
        
        ins_att_score_q, _ = query_query1.max(-1)  # (B * total_Q, L)
        #[40,128]
        ins_att_score_q=torch.div(ins_att_score_q,math.sqrt(self.hidden_size))
        query_query_norm=torch.div(query_query, math.sqrt(self.hidden_size))#[10,128,128] 
        query_mask=query['mask'].unsqueeze(1)&(self.a+self.c)#[10,128,128]
        ins_att_score_q =self.phase_attn(query_query_norm,query_mask,ins_att_score_q, query['mask'],query_loc)
        ins_att_score_q =  ins_att_score_q.unsqueeze(-1)  # (B * total_Q, L, 1)
        #[40,128,1]
        query_loc = torch.sum(ins_att_score_q * query_loc, dim=1)  # (B * total_Q, D)
        #[40,768]
        query_loc = query_loc.view(B, total_Q, self.hidden_size)  # (B, total_Q, D)
        #[4,10,768]
        
        # local prototypes  (可以先通过傅里叶变换进行传递)
        ###傅里叶
        support_loc=torch.mean(support_loc, 2).unsqueeze(2)+rel_text_loc.unsqueeze(2)#[4,10,1,768]????????
        support_loc1=self.gcn(support_loc,N,1,B) #[4,10,1,768]
        
        ###
        proto_loc = torch.mean(support_loc, 2) + rel_text_loc  # (B, N, D)
        #[4,10,768]
        proto_loc1 = torch.mean(support_loc1, 2) + rel_text_loc  # (B, N, D)
        # hybrid prototype
        proto_hyb = torch.cat((proto_glo, proto_loc), dim=-1)  # (B, N, 3D)
        # [4,10,2304]
        proto_hyb1 = torch.cat((proto_glo, proto_loc1), dim=-1)  # (B, N, 3D)
        query_hyb = torch.cat((query_glo, query_loc), dim=-1)  # (B, total_Q, 3D)
        #[4,10,2304]
        rel_text_hyb = torch.cat((rel_text_glo, rel_text_loc), dim=-1)  # (B, N, 3D)
        #[4, 10, 2304]
        
       
       
       
       
        logits = self.__batch_dist__(proto_hyb1, query_hyb)  # (B, total_Q, N)
       
       
        #logits = self.__batch_dist__(proto_loc1, query_loc)
       
       
       
       #[4,10,10]
        minn, _ = logits.min(-1) #[4,10]
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        #[4,10,11]
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits_proto, labels_proto, sim_scalar = None, None, None

        if not is_eval:
            # relation-prototype contrastive learning
            # # # relation as anchor
            rel_text_anchor = rel_text_hyb.view(B * N, -1).unsqueeze(1)  # (B * N, 1, 3D)
            #[40, 1, 2304]
            
            # select positive prototypes
            proto_hyb = proto_hyb.view(B * N, -1)  # (B * N, 3D)
            #[40,2304]
            pos_proto_hyb = proto_hyb.unsqueeze(1)  # (B * N, 1, 3D)
            #[40,1,2304]
            # select negative prototypes
            neg_index = torch.zeros(B, N, N - 1)  # (B, N, N - 1)
            #[4,10,9]
            for b in range(B):#4
                for i in range(N):#10
                    index_ori = [i for i in range(b * N, (b + 1) * N)]
                    index_ori.pop(i)
                    neg_index[b, i] = torch.tensor(index_ori)
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            neg_index = neg_index.long().view(-1).cuda()  # (B * N * (N - 1))
            #[360]
            neg_proto_hyb = torch.index_select(proto_hyb, dim=0, index=neg_index).view(B * N, N - 1, -1)
            #[40, 9 , 2304]
            # compute prototypical logits
            proto_selected = torch.cat((pos_proto_hyb, neg_proto_hyb), dim=1)  # (B * N, N, 3D)
            #[40,10,2304]
            
            logits_proto = self.__batch_dist__(proto_selected, rel_text_anchor).squeeze(1)  # (B * N, N)
            #[10,1,10]
            logits_proto /= self.temp_proto  # scaling temperatures for the selected prototypes

            # targets
            labels_proto = torch.cat((torch.ones(B * N, 1), torch.zeros(B * N, N - 1)), dim=-1).cuda()  # (B * N, 2N)
            
            # task similarity scalar
            features_sim = torch.cat((proto_hyb1.view(B, N, -1), rel_text_hyb), dim=-1)
            #[4,10,4608]
            features_sim = self.l2norm(features_sim)
            sim_task = torch.bmm(features_sim, torch.transpose(features_sim, 2, 1))  # (B, N, N)
            #[4,10,10]
            sim_scalar = torch.norm(sim_task, dim=(1, 2))  # (B) [4]
            sim_scalar = torch.softmax(sim_scalar, dim=-1)
            sim_scalar = sim_scalar.repeat(total_Q, 1).t().reshape(-1)  # (B*totalQ)
            #[40]
        return logits, pred, logits_proto, labels_proto, sim_scalar,0


    def phase_attn(self,input_score1, mask1, input_score2, mask2, input_loc):    
        #连续性计算
        constituent=self.compute_constituent(input_score1, mask1) #[40,128,128]
        input_score3= input_score2.masked_fill(mask2==0, 0)
        seg_v, seg_i = input_score3.max(-1)#[40]    
        
        #1)计算通用虚词att  负相关
        general_phase_score = -torch.bmm(input_loc,self.weight_word.unsqueeze(0).repeat(input_loc.size(0),1,1)).squeeze(-1)#[10,128]
        #[10,128]

        #2)连续性
        cons_phase_score=torch.zeros(constituent.size(0),self.max_len).cuda()
        #[10,128] 
        for i in range(constituent.size(0)):
            d= torch.index_select(constituent[i], dim = 0, index =seg_i[i]) #[5,128]
            cons_phase_score[i]=d
            if seg_i[i]==0:
                cons_phase_score[i][seg_i[i]]+=d[0][seg_i[i]+1]
            elif seg_i[i]==self.max_len-1:
                cons_phase_score[i][seg_i[i]]+=d[0][seg_i[i]-1]               
            else:
                cons_phase_score[i][seg_i[i]]+=torch.max(d[0][seg_i[i]-1]+d[0][seg_i[i]+1])

        #3)related
        related_phase_score =torch.zeros(constituent.size(0),self.max_len).cuda()
        #[10,5,128]
        for i in range(related_phase_score.size(0)):
            related_phase_score[i]= torch.index_select(input_score1[i], dim = 0, index =seg_i[i])#[5,128]
            related_phase_score[i][seg_i[i]]=0
            related_phase_score[i][seg_i[i]],_=related_phase_score[i].max(-1)
        
        related_phase_score = self.l2norm(related_phase_score)
        seg_score=F.softmax(torch.tanh(input_score2), dim=1)
        cons_phase_score=F.softmax(cons_phase_score, dim=1)
        related_phase_score=F.softmax(related_phase_score,dim=1)#[10,128,128]
        general_phase_score  = F.softmax( general_phase_score,dim=1)
        #[10,5,128]        
       # general_phase_score=self.dropout(general_phase_score)
        final_phase_attn = 0.6*seg_score++0.2*general_phase_score+0.2*related_phase_score#0.3*cons_phase_score +0.1*general_phase_score+0.1*related_phase_score#+0.2*general_phase_score#[10,768]#???权重调整
        #[ 40,128]
        final_phase_attn=self.dropout(final_phase_attn)
        return final_phase_attn     


    def gcn(self, support_loc, N, K, B):
        #support_loc [4,10,1,768]
        #初始化邻接矩阵
        full_edge,full_edge1=self.set_init_edge(support_loc,B,N,K)#[B,2,N*K,N*K]  0层为同类 1层为不同类
        #[1,1,10,10]
        support_loc=support_loc.view(B,N*K,self.hidden_size)#[1,10,768]
        if K>1:
            node_feat1=self.node_update(support_loc,full_edge[:,0],islow=True)#低频
        node_feat2=self.node_update(support_loc,full_edge[:,0],full_edge1,islow=False)#高频
        if K>1:
            support_loc=support_loc+node_feat1+ node_feat2
        else:
            support_loc= support_loc+node_feat2#?????????????????
        support_loc=support_loc.view(B,N,K,self.hidden_size)
        return support_loc  #[4,10,1,768]
        
        
    def node_update(self, node_feat,edge_feat,edge_feat1,islow):
        """input: 
                node_feat:[B,node_num,D] [1,20,768]
                edge_feat:[B,node_num,node_num] [1,20,20]
        """        
        laplacian=self.get_laplacian(edge_feat,edge_feat1,normalized=True)#[1,10,10]
        cur_node_feat = self.chebyshev(node_feat,laplacian,islow)#[1,20,130]  batch_size x node_num x out_features
        return cur_node_feat#[1,20,90] [B,N*K,D]   
    
    def chebyshev(self, x, laplacian, islow):#x [1,20,768]  laplacian [1，20，20]
        """ input:
               x:node_feat [1,20,768]
              laplacian: [1,20,20]
              islow: same is true ; fasle is different
        """
        batch_size, node_num, in_features = x.size()#1 20 768
        x0=x.float()#[1,20,768]
        x0 = self.dropout(x0)
        
        if islow:  
            cur_low_filter= ((self.eps+1)* torch.eye(node_num)).unsqueeze(0)-laplacian#[1,20,20]
            cur_low_filter=cur_low_filter.cuda()
            filter_x1=torch.bmm(cur_low_filter, x0) #[1,20,768]
        else:     
            cur_high_filter= torch.eye(node_num).unsqueeze(0) +laplacian
            cur_high_filter=cur_high_filter.cuda()
           # cur_high_filter=self.dropout(cur_high_filter)#???????????????
            filter_x1=torch.bmm(cur_high_filter, x0) #[1,20,768]
            
        return filter_x1#[1,20,130]
     
    def get_laplacian(self, W , W1, normalized=True):
        B,N1,N2=W.shape#1 10 10
        L = torch.zeros(B,N1,N2)#[1,20,20]
        pos_W1=torch.abs(W1)
        for i in range(B):
            # Degree matrix.
            #2)角度
            d1 = pos_W1[i].sum(axis=1)#[20]
            d = 1 / torch.sqrt(d1)#[20]
            #3)Normalized Laplacian matrix.
            D=torch.diag(d).cuda()   
            Li = torch.mm(torch.mm(D , W[i]), D)
            L[i]=Li
        return L 
        
    def set_init_edge(self,input_loc,B,N,K):
        '''
        B,N,K:构建初始的邻接矩阵
        '''
        #[[0, 0, 1, 1]]#[B,N*K]
        #[[0, 1]]#[B,Q]
        input_loc=input_loc.view(B,N*K,-1)
        score=torch.bmm(input_loc,torch.transpose(input_loc, -1, -2))
        score=torch.div(score,math.sqrt(self.hidden_size))#[1,10,10]
        score=self.l2norm(score)#[1,10,10]
        start=1
        full_label=[]
        for i in range(N):
            for j in range(K):
                full_label.append(start)
            start+=1
        full_label=torch.tensor(full_label).unsqueeze(0).repeat(B,1)#[B,N*K]
        #[[0, 0, 1, 1, 0, 1]]
        # get size
        num_samples = full_label.size(1)# 10
        # reshape
        label_i = full_label.unsqueeze(-1).repeat(1, 1, num_samples)#[40,10,10]
        label_j = label_i.transpose(1, 2)
       
        # compute edge
        edge_mask = 1-torch.eq(label_i, label_j).float()#[1,10,10]
        edge_mask =edge_mask.cuda()#[40,10,10]
        score1=edge_mask*score
        
        
        score_mean=score1.sum(-1)/(N*K-K)#[1,10]
        score_mean=score_mean.unsqueeze(-1).repeat(1,1,N*K)
        score=score_mean-score1
        score=torch.clamp(score,max=0.0).masked_fill(edge_mask==0,0.0)
        # expand
        score = score.unsqueeze(1)#[40.1.10.10]
        
        return score,score1#[1,2,6,6]

        
