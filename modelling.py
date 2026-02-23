import math
from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
    
class EncoderForSeqRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_id = args.model_id if args.benchmark and \
            not (Path(args.model_name_or_path) / 'model.safetensors').exists() \
            else args.model_name_or_path
        self.model = AutoModel.from_pretrained(
            model_id,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype
        )

        assert not args.attn_implementation == "flash_attention_2" \
            or self.model.config._attn_implementation == "flash_attention_2"
        if args.lora: self.model = setup_lora(self.model, args)
        self.encoder_method = args.encoder_method
        self.use_item_table = args.use_item_table
        self.train_num_candidates = args.train_num_candidates_per_row
        self.train_num_history_items = args.max_item_embeddings
        self.train_batch_size = args.train_batch_size
        self.loss = args.loss
        self.uniformity_lam = args.uniformity_lam
        self.contrast_margin = args.contrast_margin
        self.similarity_temp = Temperature(init_tau=args.temp, freeze=args.freeze_temp) if args.loss in ['infonce', 'bce_temp', 'multi_infonce', 'ranknet'] else None
        self.min_negative = torch.finfo(args.dtype).min #-1e9
        hidden_size = self.model.config.hidden_size
        self.token_pooler = Pooler(pooler_type=args.token_pooling, padding_side=args.padding_side)
        self.item_pooler = SASRecPooler(d_in=hidden_size, d_model=256) if args.item_pooling == 'sas' \
            else Pooler(pooler_type=args.item_pooling, padding_side=args.padding_side, normalize=args.item_pool_prenorm)
        self.score_head = ScoringHead(hidden_size, args) if args.use_score_head else None
        self.score_mix_lam = args.score_mix_lam if args.encoder_method == 'whiten_mix_bi' else None
        self.score_redun_lam = args.score_redun_lam
        self.score_redun_temp = self.similarity_temp
        self.redun_sim_thresh = args.redun_sim_thresh
        self.mean_pooler = Pooler(pooler_type='mean') if args.encoder_method == 'whiten_mix_bi' else None
        self.centroid_pooler = None
        if args.item_pooling == 'topk':
            self.centroid_pooler = TopKPooler(k=args.pooling_topk)
        
    def init_item_embedding(self, embeddings: Optional[torch.Tensor] = None, freeze=False):
        if embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze, padding_idx=0)
            print('Initalized item embeddings from vectors.')
        
    def forward(self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        candidate_item_mask: Optional[torch.Tensor] = None, # bi-encoder train only
        candidate_group_length: List[int] = None, # bi-encoder test only
        labels: Optional[torch.Tensor] = None,
        # bi-encoder-only kwargs
        history_item_mask: Optional[torch.Tensor] = None,
        num_history_items: int = None,
        # cross-encoder-only kwargs
        candidate_input_ids: Optional[torch.Tensor] = None,
        candidate_attention_mask: Optional[torch.Tensor] = None,
        # cross-encoder with embedding indexing
        candidate_ids: list[int] = None,
    ):
        batch_metrics = None
        hist_emb, hist_item_emb, hist_item_mask = None, None, None
        cand_item_emb, cand_item_mask = None, None
        
        if self.encoder_method == 'bi':
            if not self.use_item_table:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                flat_embeddings = self.token_pooler(output.last_hidden_state, attention_mask)
            else:
                flat_embeddings = self.item_embedding(input_ids)
            
            hist_item_emb, hist_item_mask = self.rearrange_batch_items(
                flat_embeddings=flat_embeddings[:num_history_items, :],
                item_mask=history_item_mask,
                group_length=None,# history_group_length,
            )
            hist_emb = self.item_pooler(hist_item_emb, hist_item_mask)
            
            cand_item_emb, cand_item_mask = self.rearrange_batch_items(
                flat_embeddings=flat_embeddings[num_history_items:, :],
                item_mask=candidate_item_mask,
                group_length=candidate_group_length,
            )
            logits = self.compute_scores(hist_emb, cand_item_emb)
            batch_metrics = self.gather_metrics(hist_emb, cand_item_emb, cand_item_mask, logits)
        
        elif self.encoder_method == 'cross' and self.use_item_table:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hist_emb = self.token_pooler(output.last_hidden_state, attention_mask)
            
            flat_embeddings = self.item_embedding(candidate_ids)
            cand_item_emb, cand_item_mask = self.rearrange_batch_items(
                flat_embeddings=flat_embeddings,
                item_mask=candidate_item_mask,
                group_length=candidate_group_length,
            )
            logits = self.compute_scores(hist_emb, cand_item_emb) 
            batch_metrics = self.gather_metrics(hist_emb, cand_item_emb, cand_item_mask, logits)

        elif self.encoder_method == 'cross':
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hist_emb = self.token_pooler(output.last_hidden_state, attention_mask)
            
            candidate_output = self.model(input_ids=candidate_input_ids, attention_mask=candidate_attention_mask)
            flat_embeddings = self.token_pooler(candidate_output.last_hidden_state, candidate_attention_mask)
            cand_item_emb, cand_item_mask = self.rearrange_batch_items(
                flat_embeddings=flat_embeddings,
                item_mask=candidate_item_mask,
                group_length=candidate_group_length,
            )
            logits = self.compute_scores(hist_emb, cand_item_emb) 
            batch_metrics = self.gather_metrics(hist_emb, cand_item_emb, cand_item_mask, logits)
            
        elif self.encoder_method == 'bi_colbert':
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            flat_embeddings = output.last_hidden_state # b*k,l,h
            b, k, l, h = history_item_mask.shape[0], history_item_mask.shape[1], flat_embeddings.shape[1], flat_embeddings.shape[-1]
            
            # flattened history tokens across all history items
            hist_item_emb = flat_embeddings[:num_history_items].view((b, k*l, h))
            hist_attention_mask = ( # merge by broadcasting item mask then flattening
                attention_mask[:num_history_items].view((b, k, l)).bool() & history_item_mask.view((b, k, 1)).bool()
            ).view((b, k*l))
            
            # candidates sort to b,n,l,h; b,n,l for token attention mask
            if labels is not None:
                n = candidate_item_mask.shape[1]
                cand_item_emb = flat_embeddings[num_history_items:].view((b, n, l, h))
                cand_attention_mask = (
                    attention_mask[num_history_items:].view((b, n, l)).bool() & candidate_item_mask.view((b, n, 1)).bool()
                )
                cand_item_mask = candidate_item_mask
            else: # ragged
                n = max(candidate_group_length)
                cand_item_emb, cand_item_mask = self._regroup(
                    flat_embeddings[num_history_items:], candidate_group_length, shape=(b, n, l, h)
                )
                cand_attention_mask = self._merge_to_token_mask(attention_mask[num_history_items:], cand_item_mask, candidate_group_length)
            
            logits = self.compute_late_interaction(
                hist_item_emb, hist_attention_mask, cand_item_emb, cand_attention_mask, 
            )
        elif self.encoder_method == 'cross_colbert':
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hist_emb = output.last_hidden_state # b,l,h
            cand_item_emb = self.model(input_ids=candidate_input_ids, attention_mask=candidate_attention_mask).last_hidden_state
            b, l, h = input_ids.shape[0], input_ids.shape[1], hist_emb.shape[-1]
            
            # candidates sort to b,n,l,h; b,n,l for token attention mask
            if labels is not None:
                n, c_l = candidate_item_mask.shape[1], candidate_input_ids.shape[1]
                cand_item_emb = cand_item_emb.view((b, n, c_l, h))
                cand_item_mask = candidate_item_mask
                cand_attention_mask = candidate_attention_mask.view((b, n, c_l)).bool() \
                                        & candidate_item_mask.view((b, n, 1)).bool()
            else: # ragged
                n, c_l = max(candidate_group_length), candidate_input_ids.shape[1]
                cand_item_emb, cand_item_mask = self._regroup(
                    cand_item_emb, candidate_group_length, shape=(b, n, c_l, h)
                )
                cand_attention_mask = self._merge_to_token_mask(candidate_attention_mask, cand_item_mask, candidate_group_length)
            
            logits = self.compute_late_interaction(
                hist_emb, attention_mask, cand_item_emb, cand_attention_mask, 
            )
        elif self.encoder_method == 'full_cross':
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hist_emb = self.token_pooler(output.last_hidden_state, attention_mask)
            b, l, h = input_ids.shape[0], input_ids.shape[1], hist_emb.shape[-1]
            
            logits = self.compute_scores(hist_emb, None) 
            batch_metrics = self.gather_metrics(hist_emb, None, None, logits)
            
            if labels is None: return logits, batch_metrics
            # only pass to loss function for masking not score head
            cand_item_mask = candidate_item_mask 
            logits = logits.view(candidate_item_mask.shape)
        elif self.encoder_method == 'bi_mix_topk_mean':
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            flat_embeddings = self.token_pooler(output.last_hidden_state, attention_mask)            
            hist_item_emb, hist_item_mask = self.rearrange_batch_items(
                flat_embeddings=flat_embeddings[:num_history_items, :],
                item_mask=history_item_mask,
                group_length=None,
            )
            cand_item_emb, cand_item_mask = self.rearrange_batch_items(
                flat_embeddings=flat_embeddings[num_history_items:, :],
                item_mask=candidate_item_mask,
                group_length=candidate_group_length,
            )
            
            base_score, hist_emb_k = self.centroid_pooler(
                hist_item_emb, hist_item_mask, cand_item_emb, temperature=self.similarity_temp()
            )
            # logits = base_score
            hist_emb = self.mean_pooler(hist_item_emb, hist_item_mask)
            logits = self.score_mix_lam * self.compute_scores(hist_emb, cand_item_emb) \
                     + (1 - self.score_mix_lam) * base_score
            batch_metrics = self.gather_metrics(hist_emb, cand_item_emb, cand_item_mask, logits)
            
        else: raise NotImplementedError
        if labels is None: return logits[cand_item_mask.bool()], batch_metrics
        if self.score_redun_lam: logits = self._add_mrr(logits, hist_item_emb, hist_item_mask, cand_item_emb)
        
        loss = self.compute_loss(logits, cand_item_mask, labels)
        # additional loss terms
        if self.uniformity_lam: # colbert no hist_item_mask for hist_item_emb yet
            if hist_emb is None: hist_emb = hist_item_emb[hist_item_mask.bool()]
            scores = self._calculate_uniformity(F.normalize(cand_item_emb[cand_item_mask.bool()], dim=-1), reduce=True)
            # + self._calculate_uniformity(F.normalize(hist_emb, dim=-1), reduce=True) \
            loss = loss + self.uniformity_lam * scores
            
        return loss
    
    def compute_loss(self, logits, cand_item_mask, labels):
        '''(b, k) candidates with (b, k) arbitary labels'''
        loss = None
        if self.loss == 'infonce':
            ignore_mask = (1 - cand_item_mask).bool()
            logits = logits.masked_fill(ignore_mask, self.min_negative)
            loss = F.cross_entropy(logits, labels.long().argmax(dim=-1), reduction='mean')
            loss = loss - (-cand_item_mask.sum(-1).float().log().mean()) # only affects logging
        elif self.loss == 'multi_infonce':
            ignore_mask = (1 - cand_item_mask).bool()
            pos = labels.bool() & cand_item_mask.bool()

            # --- "better" sup-out variant with log inside positive summation
            logits = logits.masked_fill(ignore_mask, float("-inf"))
            logp = F.log_softmax(logits, dim=-1)

            # mean log-prob over positives per anchor
            loss_i = -(logp.masked_fill(~pos, 0.0).sum(dim=-1) / pos.sum(dim=-1))
            loss = loss_i.mean()

            # --- "worse" sup-in variant with log outside positive summation (more intuitive)
            # denom = torch.logsumexp(logits.masked_fill(ignore_mask, self.min_negative), dim=-1)
            # numer = torch.logsumexp(logits.masked_fill(~pos, self.min_negative), dim=-1) - pos.sum(1).log()
            # loss = (- numer + denom).mean()
        elif self.loss == 'ranknet':
            pos_mask = labels * cand_item_mask.bool()
            neg_mask = (1 - labels) * cand_item_mask.bool()
            
            diffs = logits.unsqueeze(2) - logits.unsqueeze(1)
            pair_mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
            
            loss = F.binary_cross_entropy_with_logits(
                diffs, torch.ones_like(diffs), reduction='none'
            )
            loss = (loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)
        elif self.loss == 'margin_rank': # pairwise contrastive pos > neg
            pos_mask = labels * cand_item_mask.bool()
            neg_mask = (1 - labels) * cand_item_mask.bool()
            
            diffs = logits.unsqueeze(2) - logits.unsqueeze(1)
            pair_mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
            loss = F.relu(self.contrast_margin - diffs)
            loss = (loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)
        elif self.loss == 'bce_temp':
            ignore_mask = (1 - cand_item_mask).bool()
            logits = logits.masked_fill(ignore_mask, self.min_negative)
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), reduction='mean')
        elif self.loss == 'cos_emb': # torch.CosineEmbeddingLoss but require history broadcast
            pos_mask = labels * cand_item_mask.bool()
            neg_mask = (1 - labels) * cand_item_mask.bool()
            
            loss = ((1 - logits) * pos_mask).sum() \
                    + ((logits - self.contrast_margin) * neg_mask).sum()
            
            loss = loss / cand_item_mask.sum().clamp(min=1)
        else:
            raise NotImplementedError
        
        return loss
        
    def _regroup(self, t, group_len, shape=None):
        '''Reshape flattened 2D embeddings to (b, n, h) tensor'''
        b, max_c = len(group_len), max(group_len)
        if shape is None: shape = (b, max_c, t.size(-1))
        item_padded = t.new_zeros(shape)
        item_mask = torch.zeros((b, max_c), dtype=torch.bool, device=t.device)
        offset = 0
        for i, c_i in enumerate(group_len):
            item_padded[i, :c_i] = t[offset:offset+c_i]
            item_mask[i, :c_i] = True
            offset = offset + c_i
        return item_padded, item_mask # padded [(b,n,h) | shape] ; item mask (b,n)

    def _merge_to_token_mask(self, token_mask, item_mask, group_len):
        b, n = item_mask.shape
        l = token_mask.shape[1]
        mask = torch.zeros((b, n, l), device=item_mask.device)
        offset = 0
        for i, c_i in enumerate(group_len):
            mask[i, :c_i] = token_mask[offset:offset+c_i]
            offset = offset + c_i
        return mask.bool() & item_mask.view((b, n, 1)).bool()
        
    def rearrange_batch_items(
        self,
        flat_embeddings: Optional[torch.Tensor] = None,
        item_mask: Optional[torch.Tensor] = None,
        group_length: Union[int, List[int], None] = None,
    ):
        if group_length is None:
            shape = (item_mask.shape[0], item_mask.shape[1], flat_embeddings.shape[-1])
            item_padded = flat_embeddings.view(shape)
        else:
            item_padded, item_mask = self._regroup(flat_embeddings, group_length)
        return item_padded, item_mask # b,n,h
    
    def compute_scores(self, hist_emb, cand_item_emb):
        if self.score_head:
            scores = self.score_head(hist_emb.unsqueeze(1), cand_item_emb)
        else:
            scores = F.cosine_similarity(hist_emb.unsqueeze(1), cand_item_emb, dim=-1)
        if self.similarity_temp is not None: scores = scores / self.similarity_temp()
        return scores
    
    def compute_late_interaction(self, hist_item_emb, hist_attention_mask, cand_item_emb, cand_attention_mask):
        hist_item_emb = F.normalize(hist_item_emb, p=2, dim=-1).unsqueeze(1) # b, 1, k * L1, h
        cand_item_emb = F.normalize(cand_item_emb, p=2, dim=-1) # b, n, L2, h
        max_sim = hist_item_emb @ cand_item_emb.transpose(-2, -1)
        
        cand_mask = cand_attention_mask.bool().unsqueeze(2)
        max_sim = max_sim.masked_fill(~cand_mask, self.min_negative)
        max_sim = (max_sim / 0.1).logsumexp(dim=-1) * 0.1
        
        hist_mask = hist_attention_mask.bool().unsqueeze(1)
        max_sim = max_sim.masked_fill(~hist_mask, 0.0)
        denom = hist_attention_mask.sum(dim=-1).clamp_min(1).unsqueeze(1)  # b,1
        max_sim = max_sim.sum(dim=-1) / denom # b,n
        return max_sim
    
    def encode_items(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.token_pooler(output.last_hidden_state, attention_mask)
        return embeddings
    
    # ---------- training metric logging ----------
    def gather_metrics(self, hist_emb, cand_item_emb, cand_item_mask, scores):
        if self.training: return None
        if cand_item_mask is not None: 
            cand_item_emb = cand_item_emb[cand_item_mask.bool()]
            scores = scores[cand_item_mask.bool()]
        norm_hist_emb = nn.functional.normalize(hist_emb, p=2, dim=-1) 
        metrics = {
            'score_count': scores.numel(),
            'cos_score': scores.view((-1)).sum().item(),
            'hist_count': hist_emb.shape[0],
            'hist_norm': torch.norm(hist_emb, dim=1).sum().item(),
            'hist_cos': self._calculate_pairwise_cosine(norm_hist_emb).item(),
            'hist_uni': self._calculate_uniformity(norm_hist_emb).item(),
            'hist_emb': hist_emb.cpu()
        }
        if cand_item_emb is not None:
            norm_cand_item_emb = nn.functional.normalize(cand_item_emb, p=2, dim=-1) 
            metrics['cand_count'] = cand_item_emb.shape[0]
            metrics['cand_norm'] = torch.norm(cand_item_emb, dim=1).view((-1)).sum().item()
            metrics['cand_cos'] = self._calculate_pairwise_cosine(norm_cand_item_emb).item()
            metrics['cand_uni'] = self._calculate_uniformity(norm_cand_item_emb).item()
            metrics['cand_emb'] = cand_item_emb.view((-1, cand_item_emb.shape[-1])).cpu()
        return metrics
    
    def _calculate_pairwise_cosine(self, norm_emb):
        sim_matrix = norm_emb @ norm_emb.transpose(-2, -1)
        n_diag = float(sim_matrix.shape[0])
        return (sim_matrix.sum() - n_diag)
        
    def _calculate_uniformity(self, norm_emb, t=2.0, reduce=False):
        diff = 2 - 2 * norm_emb @ norm_emb.T
        # mask for removing diagonals as self-dot-product, and flattening scores
        mask = ~torch.eye(diff.shape[0], dtype=torch.bool, device=diff.device)
        if not reduce: return (-t * diff[mask]).exp().sum()
        return (-t * diff[mask]).exp().mean().log()
    
    def _add_mrr(self, scores, hist_item_emb, hist_item_mask, cand_item_emb):
        # dedup
        pair_cos = torch.matmul(
            F.normalize(cand_item_emb, p=2, dim=1),
            F.normalize(hist_item_emb, p=2, dim=-1).transpose(-2, -1)
        )
        pair_cos, _ = pair_cos.masked_fill(~hist_item_mask.bool().unsqueeze(1), -1e9).max(-1)
        redun_penalty = -((pair_cos - self.redun_sim_thresh)).clamp(min=0.0)
        scores = scores + self.score_redun_lam * redun_penalty
        return scores
    
    
###########
# POOLING #
###########
    
class Pooler(nn.Module):
    def __init__(self, pooler_type: Optional[str] = 'mean', normalize=False, padding_side=None):
        super().__init__()
        self.pooler_type = pooler_type
        self.padding_side = padding_side
        self.normalize = normalize

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = None
        if self.pooler_type == 'cls':
            output = hidden_states[:, 0]
        elif self.pooler_type == "mean":
            if self.normalize: hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            # only added clamp min to 1
            output = ((hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1).clamp(min=1.0))
        elif self.pooler_type == 'last':
            if self.padding_side == 'left':
                output = hidden_states[:, -1]
            elif self.padding_side == 'right':
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.shape[0]
                output = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else: raise NotImplementedError
        return output

class TopKPooler(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.k = k
        self.min_negative = -1e9
    
    def forward(self, hist_item_emb, hist_item_mask, cand_item_emb, temperature=None):
        # top k pool (b, n, h) with (b, m, h)
        hist_item_emb = F.normalize(hist_item_emb, p=2, dim=-1)
        cand_item_emb = F.normalize(cand_item_emb, p=2, dim=-1)
        scores = cand_item_emb @ hist_item_emb.transpose(-2, -1)
        scores = scores.masked_fill(~hist_item_mask.bool().unsqueeze(1), self.min_negative)
        top_v, top_i = torch.topk(scores, k=min(self.k, hist_item_emb.size(1)), dim=2, sorted=False)
        
        valid = top_v > self.min_negative / 2
        top_i = top_i.unsqueeze(-1).expand(-1, -1, -1, hist_item_emb.size(2)) # (b, m, k, h)
        top_k_emb = torch.gather(
            hist_item_emb.unsqueeze(1).expand(-1, cand_item_emb.size(1), -1, hist_item_emb.size(2)),
            dim=2, index=top_i
        ) # select k out of N
        
        # this calculates mean score amongst top k items
        top_k_mean_score = (top_v * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        
        # instead mean pool top k (already normalized), then score        
        top_k_mean_emb = (top_k_emb * valid.unsqueeze(-1)).sum(dim=2) / valid.sum(dim=2).unsqueeze(-1).clamp_min(1.0)
        # top_k_mean_score = F.cosine_similarity(top_k_mean_emb, cand_item_emb, dim=-1)
                
        if temperature: top_k_mean_score = top_k_mean_score / temperature
        return top_k_mean_score, top_k_mean_emb

class SASRecPooler(nn.Module):
    def __init__(
        self, d_in, d_model, max_length=64, 
        mult_d_ffn=2, nhead=4, num_layers=2, dropout=0.2
    ):
        super().__init__()
        self.max_length = max_length
        self.hidden_size = d_model
        self.position_embedding = nn.Embedding(self.max_length, d_model)
        self.proj_in = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.proj_out = nn.Linear(d_model, d_in) if d_in != d_model else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_in = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mult_d_ffn,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        self.register_buffer(
            "position_ids",
            torch.arange(max_length, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
                
        self.apply(self._init_module)
        
    def forward(self, sequence_embeddings: torch.Tensor, sequence_mask):
        b, l, _ = sequence_embeddings.shape
        x = self.proj_in(sequence_embeddings)
        
        x = x + self.position_embedding(self.position_ids[:, :l])
        
        x = self.dropout(self.layer_norm_in(x))
        
        padding_mask = ~sequence_mask.bool()
        causal_mask = torch.triu(x.new_ones((l, l)), diagonal=1).bool()
        hidden_states = self.encoder(
            x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True
        )
        
        hidden_states = self.proj_out(hidden_states)
        last_idx = sequence_mask.sum(1).clamp(min=1) - 1
        return hidden_states[torch.arange(b), last_idx.int()]
    
    def _init_module(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)
            
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)

            if getattr(m, "q_proj_weight", None) is not None:
                nn.init.xavier_uniform_(m.q_proj_weight)
            if getattr(m, "k_proj_weight", None) is not None:
                nn.init.xavier_uniform_(m.k_proj_weight)
            if getattr(m, "v_proj_weight", None) is not None:
                nn.init.xavier_uniform_(m.v_proj_weight)

            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
                
########
# MISC #
########
                
class ScoringHead(nn.Module):
    def __init__(self, hidden_dim, args):
        super().__init__()
        input_dim = hidden_dim if args.encoder_method == 'full_cross' else 3 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hist_emb: torch.Tensor, cand_emb: Optional[torch.Tensor] = None):
        if cand_emb is not None:
            b, n, h = cand_emb.shape
            hist_emb = hist_emb.expand(-1, n, -1)
            x = torch.cat([hist_emb, cand_emb, hist_emb * cand_emb], dim=-1)
            score = self.mlp(x.view(b*n, h*3))
            return score.view((b, n))
        else:
            return self.mlp(hist_emb)

class Temperature(nn.Module):
    def __init__(self, init_tau=0.1, min_tau=0.01, max_tau=2.0, freeze=False):
        super().__init__()
        # log tau to enforce positive temperature and log-scale
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)), requires_grad=freeze)
        self.min_tau = min_tau
        self.max_tau = max_tau

    def forward(self):
        tau = self.log_tau.exp()
        return tau.clamp(self.min_tau, self.max_tau)
    
################
# LORA HELPERS #
################

import re
def guess_lora_targets(
    model,
    include_attention: bool = True,
    include_mlp: bool = True,
    include_output: bool = False,   # e.g., lm_head for LMs; usually False for retrieval encoders
):
    """
    Returns a list of *leaf module names* suitable for PEFT `target_modules`.
    Works best for HF Transformer architectures with conventional naming.

    include_attention: q/k/v/o (or equivalents)
    include_mlp: FFN / MLP projections (fc1/fc2, up/down/gate, wi/wo, etc.)
    include_output: output heads (lm_head / classifier). Usually handled via modules_to_save instead.
    """

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    presets = {
        "modernbert": {
            
        },
        "llama": {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp":  ["gate_proj", "up_proj", "down_proj"],
        },
        "mistral": {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp":  ["gate_proj", "up_proj", "down_proj"],
        },
        "qwen3": {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp":  ["gate_proj", "up_proj", "down_proj"],
        },

        "qwen2": {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp":  ["gate_proj", "up_proj", "down_proj"],
        },
        "gemma": {
            "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp":  ["gate_proj", "up_proj", "down_proj"],
        },
        "bert": {
            "attn": ["query", "key", "value"],
            "mlp":  ["dense"],
        },
        "roberta": {
            "attn": ["query", "key", "value"],
            "mlp":  ["dense"],
        },
        "deberta-v2": {
            "attn": ["query_proj", "key_proj", "value_proj"],
            "mlp":  ["dense"],
        },
        "t5": {
            "attn": ["q", "k", "v", "o"],
            "mlp":  ["wi", "wo"],
        },
        "gpt2": {
            "attn": ["c_attn", "c_proj"],
            "mlp":  ["c_fc", "c_proj"],  # note: c_proj appears in attn and mlp blocks
        },
    }

    chosen = set()
    
    # ModernBERT LoRA preset (use regex to avoid "Wo" collision)
    if model_type == 'modernbert':
        return r".*\.attn\.(Wqkv|Wo)$|.*\.mlp\.(Wi|Wo)$" \
            if include_mlp else r".*\.attn\.(Wqkv|Wo)$"

    if model_type in presets:
        if include_attention: chosen.update(presets[model_type]["attn"])
        if include_mlp: chosen.update(presets[model_type]["mlp"])
        if include_output: chosen.update(["lm_head", "classifier", "score"])
        return sorted(chosen)

    # ---- 2) Fallback scan (broader, but catches weird variants) ----
    attn_leaf_candidates = {
        "q_proj","k_proj","v_proj","o_proj","out_proj",
        "query","key","value",
        "q","k","v","o",
        "query_key_value",
        "c_attn","c_proj",
    }
    mlp_leaf_candidates = {
        "gate_proj","up_proj","down_proj",
        "fc1","fc2","c_fc",
        "wi","wo",
        "dense_h_to_4h","dense_4h_to_h",
        "intermediate_dense","output_dense",  # occasional custom naming
        "dense",  # very common but ambiguous
    }
    output_leaf_candidates = {"lm_head","classifier","score"}

    # Path-based hints to reduce "dense" ambiguity
    attn_path_re = re.compile(r"(attn|attention|self_attn|selfattention)", re.IGNORECASE)
    mlp_path_re  = re.compile(r"(mlp|ffn|feed_forward|intermediate|output)", re.IGNORECASE)

    for name, module in model.named_modules():
        # LoRA in PEFT applies to supported module classes (commonly Linear; some models use fused/proxy layers).
        if not isinstance(module, nn.Linear):
            continue

        leaf = name.split(".")[-1]

        if include_attention and (
            leaf in attn_leaf_candidates or (
                leaf.endswith("_proj") and attn_path_re.search(name)
            )
        ): chosen.add(leaf)

        if include_mlp and (
            leaf in mlp_leaf_candidates or (
                leaf.endswith("_proj") and mlp_path_re.search(name)
            )
        ): chosen.add(leaf)

        if include_output and leaf in output_leaf_candidates: chosen.add(leaf)

    if not chosen and include_attention: # minimal default
        chosen.update(["query", "value"])
    return sorted(chosen)

def setup_lora(model, args):
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    if args.lora_dir: return PeftModel.from_pretrained(model, args.lora_dir)    
    target_modules = guess_lora_targets(model, include_output=args.use_score_head)
    print('LoRA chosen target modules: ', target_modules)
    peft_config = LoraConfig(
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model