
from typing import Optional, Callable, Dict, List, Any, Set, Tuple, TYPE_CHECKING
import hashlib
import numpy as np

from sentence_transformers import SentenceTransformer

class AlignmentResult:
    
    def __init__(self, seq1, seq2, **kwargs):
        self.seq1 = seq1
        self.seq2 = seq2
        
        self.seq1_embed = None
        self.seq2_embed = None
        
        self.params = kwargs
        self.rough_align_threshold = kwargs.get("rough_align_threshold",0.25)
        self.rough_alignment_data = None
        # self.rough_argmax = -1
        
        self.rough_checkpoint = False
        self.delta = 0
        self.rough_max = -1
        self.similarity_threshold = kwargs.get("similarity_threshold",0.7)
        self.similarities = []
        
        self.alignment_checkpoint = False
        self.aligned_pairs = []
        self.quality = -1
        self.quality_threshold_factor = kwargs.get("quality_threshold_factor",1.2)
        self.quality_threshold = -1
        
        self.success = False
        
    def print_report(self, verbose = False):
        
        parts =[]
        parts.append("===== Sequence alignment result report =====")
        
        if verbose:
            parts.append(f"seq1: {' '.join(map(str,self.seq1))}")
            parts.append(f"seq2: {' '.join(map(str,self.seq2))}")
        else:
            parts.append(f"seq1: {len(self.seq1)} tokens")
            parts.append(f"seq2: {len(self.seq2)} tokens")
        parts.append(f"params: {self.params}")
        
        # parts.append(f"rough argmax: {self.delta}")
        
        if self.rough_alignment_data and verbose:
            parts.append(f"rough alignment data: {' '.join(map(str,self.rough_alignment_data))}")
        
        if not self.rough_checkpoint:
            parts.append("failed to proceed past rough alignment")
            print('\n'.join(parts))
            return
        
        parts.append(f"rough alignment delta: {self.delta:0.5g}")
        parts.append(f"rough alignment maximum: {self.rough_max:0.5g} vs threshold {self.rough_align_threshold:0.3g}")
        if not self.alignment_checkpoint:
            parts.append("failed to proceed past sequence alignment")
            print('\n'.join(parts))
            return
        
        if verbose:
            parts.append(f"aligned pairs: {self.aligned_pairs}")
            parts.append(f"sequence similarities: {self.similarities}")
        else:
            parts.append(f"identified {len(self.aligned_pairs)} aligned pairs")
        
        parts.append(f"alignment quality: {self.quality:0.5g} vs threshold {self.quality_threshold:0.3g}")
        if self.success:
            parts.append("alignment was successful! these sequences are sufficiently similar")
        else:
            parts.append("alignment failed.. these sequences are too different")
            
        print('\n'.join(parts))
        

class SequenceCalculator:
    
    def __init__(self, tokenizer=None, model_name = 'all-MiniLM-L6-v2'):
        self.algnres = None
        self.tokenizer = tokenizer
        self.model = SentenceTransformer(model_name)
        self._cache = {}
    
    def token_ids_to_strs(self, token_ids):
        if not self.tokenizer:
            return []
        token_strs=[]
        for token_id in token_ids:
            token_strs.append(self.tokenizer.decode([token_id]))
        return token_strs
    
    def get_token_embedding(self, token_str):
        
        if token_str not in self._cache:
            embed = self.model.encode(token_str)
            self._cache[token_str] = embed
        else:
            embed = self._cache.get(token_str)
        
        return embed    
        
    def embed_sequence(self, sequence:List[str]):
        embeds = []
        for seq in sequence:
            embed = self.get_token_embedding(seq)
            embeds.append(embed)
        return np.array(embeds)
    
    def normalize_input(self, seq1, seq2):
        if isinstance(seq1[0], int):
            seq1 = self.token_ids_to_strs(seq1)
            seq2 = self.token_ids_to_strs(seq2)
        if isinstance(seq1[0], str):
            seq1 = self.embed_sequence(seq1)
            seq2 = self.embed_sequence(seq2)
        return seq1, seq2
    
    def compare_sequences(self, seq1:List[str], seq2:List[str], **kwargs):
        
        algnres = AlignmentResult(seq1, seq2, **kwargs)
        self.algnres = algnres
        
        seq1, seq2 = self.normalize_input(seq1, seq2)
        self.algnres.seq1_embed = seq1
        self.algnres.seq2_embed = seq2
        
        if len(seq2) > len(seq1):
            seq1,seq2 = seq2, seq1
        
        thresh1 = algnres.rough_align_threshold
        delta, _max = self.rough_align(seq1, seq2, thresh1) # idk what threshold to expect.. 

        if delta is None:
            return algnres
        
        algnres.delta = delta
        algnres.rough_max = _max
        algnres.rough_checkpoint=True
        
        thresh2 = algnres.similarity_threshold
        iters, quality = self.walk_sequences(seq1, seq2, delta, threshold = thresh2)
        algnres.aligned_pairs = iters
        algnres.quality = quality
        algnres.alignment_checkpoint=True
        
        thresh_factor = algnres.quality_threshold_factor
        thresh3 = thresh_factor * _max
        algnres.quality_threshold = thresh3
        
        algnres.success = quality > thresh3
        
        return algnres
    
    def align_sequences(self, seq1, seq2, aligned_pairs):
        
        fill = 0
        
        i0=j0=0
        
        newseq1 = []
        newseq2 = []
        
        aligned_pairs.append((len(seq1), len(seq2)))
        
        for i,j in aligned_pairs:
            
            subseq1 = seq1[i0:i]
            subseq2 = seq2[j0:j]
            
            delta = len(subseq1) - len(subseq2)
            
            if delta<0:
                # subseq2 is longer, fill in subseq1
                newseq1.extend([fill]*abs(delta))
            elif delta > 0:
                newseq2.extend([fill]*abs(delta))
            else:
                pass
                # theyre just fine
            
            newseq1.extend(subseq1)
            newseq2.extend(subseq2)
            
            i0,j0=i,j
        
        return newseq1, newseq2
    
    def walk_sequences(self, seq1, seq2, delta, lookahead = 5, threshold = 0.7):
        """
        basically starting from delta, give each sequence independent iterators, iterate seq1 until sequence similarity (measured via dot) about +/-lookahead is above threshold (or local maximum?), then step seq2, storing pair of iterators that appears to locally align sequences. then, do the same in the reverse. return the iterator pairs and the inner product of aligned sequences
        quality is just sum of inner product about aligned regions?
        """

        # Starting positions based on delta alignment
        if delta is None:
            return None, None
        if delta >= 0:
            i1, i2 = delta, 0
        else:
            i1, i2 = 0, -delta

        aligned_pairs = []
        sims = []
        
        # Forward walk
        while i1 < len(seq1) - lookahead and i2 < len(seq2) - lookahead:
            # Get windows for comparison
            window1 = seq1[i1:i1+lookahead]
            window2 = seq2[i2:i2+lookahead]

            # Check similarity
            # similarity = self.dot_eq(window1, window2)
            similarity = self.dot_embed(window1, window2)
            sims.append(similarity)
            # print(f"comparing sequences with similarity {similarity}, vs threshold {threshold}")
            
            if similarity >= threshold:
                # Good alignment, record and move both
                aligned_pairs.append((i1, i2))
                i1 += 1
                i2 += 1
            else:
                # Try advancing seq1
                best_sim = similarity
                best_offset = 0

                for offset in range(1, lookahead):
                    if i1 + offset < len(seq1) - lookahead:
                        test_window1 = seq1[i1+offset:i1+offset+lookahead]
                        # test_sim = self.dot_eq(test_window1, window2)
                        test_sim = self.dot_embed(test_window1, window2)
                        if test_sim > best_sim:
                            best_sim = test_sim
                            best_offset = offset

                # Try advancing seq2
                for offset in range(1, lookahead):
                    if i2 + offset < len(seq2) - lookahead:
                        test_window2 = seq2[i2+offset:i2+offset+lookahead]
                        # test_sim = self.dot_eq(window1, test_window2)
                        test_sim = self.dot_embed(window1, test_window2)
                        if test_sim > best_sim:
                            best_sim = test_sim
                            best_offset = -offset

                if best_offset > 0:
                    i1 += best_offset
                elif best_offset < 0:
                    i2 += -best_offset
                else:
                    # No improvement, advance both
                    i1 += 1
                    i2 += 1
        self.algnres.similarities = sims
        # Calculate quality as average inner product over aligned regions
        if aligned_pairs:
            # quality = sum(self.dot(seq1[i1:i1+lookahead], seq2[i2:i2+lookahead])
                        #  for i1, i2 in aligned_pairs if i1+lookahead <= len(seq1) and i2+lookahead <= len(seq2))
            quality = sum(self.dot_embed(seq1[i1:i1+lookahead], seq2[i2:i2+lookahead])
                         for i1, i2 in aligned_pairs if i1+lookahead <= len(seq1) and i2+lookahead <= len(seq2))
                        
            quality /= len(aligned_pairs)
        else:
            quality = 0

        return aligned_pairs, quality
    
    def rough_align(self, seq1, seq2, min_thresh, standardize = False):

        k_start = -len(seq1)//2
        k_end = k_start + len(seq1)
        
        # skip standardization..
        if standardize:
            seq1 = self.standardize_seq(seq1)
            seq2 = self.standardize_seq(seq2)
            factor = len(seq1)**2
        else:
            factor = sum(seq1)*sum(seq2)

        ips, argmax = self.convolve(seq1, seq2, k_start, k_end, factor=factor)
        
        self.algnres.rough_alignment_data = ips
        
        max_ip = ips[argmax]
        mean_ip = sum(ips) / len(ips) if ips else 0
        
        print(f"rough algn sbr: {max_ip / mean_ip:0.5g}")
        
        if max_ip/mean_ip > 1.2 or max_ip > min_thresh:
            return argmax + k_start, max_ip  # Return actual offset, not index
        else:
            print(f"sequences have max inner product of {max_ip}, below threshold of {min_thresh}")
            return None, -1
    
    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    def dot_embed(self, seq1, seq2):
        seq1 = seq1[:len(seq2)]
        # if not factor:
        #     factor = len(seq1) * len(seq1)
        return sum([self.cosine_similarity(a,b) for a,b in zip(seq1, seq2)])
    
    def dot(self, seq1, seq2, factor=None):
        """
        already subsequenced
        """
        seq1 = seq1[:len(seq2)]
        if not factor:
            factor = len(seq1) * len(seq1)
        return sum([a*b for a,b in zip(seq1, seq2)]) / factor
    
    def dot_eq(self, seq1, seq2):
        seq1 = seq1[:len(seq2)]
        return sum([a==b for a,b in zip(seq1, seq2)]) / len(seq1)
    
    def standardize_seq(self, seq):
        """
        standardizing sequences of tokens doesnt totally make sense .. lets see
        """
        if not seq:
            return seq

        # Calculate mean
        mean = sum(seq) / len(seq)

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in seq) / len(seq)
        std = variance ** 0.5

        if std == 0:
            return [0] * len(seq)

        # Standardize
        return [(x - mean) / std for x in seq]
    
    def convolve(self, seq1, seq2, start, end, factor = None):
        """
        Convolve seq2 across seq1, computing inner products at each offset.
        Returns list of inner products and index of maximum.
        """
        res = []
        maxval = float('-inf')
        argmax = 0

        for k in range(start, end):
            # Calculate overlap region for this offset
            # k is the offset of seq2 relative to seq1
            if k >= 0:
                # seq2 starts k positions into seq1
                start1 = k
                end1 = min(len(seq1), k + len(seq2))
                start2 = 0
                end2 = min(len(seq2), len(seq1) - k)
            else:
                # seq2 starts before seq1
                start1 = 0
                end1 = min(len(seq1), len(seq2) + k)
                start2 = -k
                end2 = min(len(seq2), len(seq1) - k)

            if end1 - start1 <= 0:
                res.append(0)
                continue

            subseq1 = seq1[start1:end1]
            subseq2 = seq2[start2:end2]

            if len(subseq1) > 0 and len(subseq1) == len(subseq2):
                # inner_prod = self.dot(subseq1, subseq2, factor=factor)
                inner_prod = self.dot_embed(subseq1, subseq2)
            else:
                inner_prod = 0

            res.append(inner_prod)

            if inner_prod > maxval:
                maxval = inner_prod
                argmax = len(res) - 1  # Index in results array
        
        return res, argmax
        