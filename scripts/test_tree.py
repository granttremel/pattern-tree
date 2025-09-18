
from pattern_tree.symbol import Symbol, tokenizer
from pattern_tree.pattern_tree import PatternTree
from pattern_tree.generator import generators as gens
from pattern_tree.zipper import SymbolZipper, ZipperState, ZipperInstruction, Move
from pattern_tree.zipper import wait_inst, search_inst, continue_inst, exit_inst, move_bits, num_moves

from pattern_tree.sequences import SequenceCalculator

import random

import json
import re
from pathlib import Path


def load_log_file(log_file:str, start=0, end = -1, formatter = None):
    
    print(f"reading log file {log_file} from {start} to {end}")
    
    log_dicts = []
    with open(log_file, 'r') as f:
        
        lines = f.readlines()
        
        # for i,line in enumerate(f):
        for i,line in enumerate(lines):
            if i < start:
                continue
            if end > start and i>=end:
                continue
            clean_line = line.strip()
            # Remove ANSI color codes
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', clean_line)
            if not clean_line:
                continue
            try:
                obj=json.loads(clean_line)
            except Exception as e:
                print(f"Error loading json line at {i}: {str(e)}")
                continue
            
            log_dicts.append(obj)
            
    if formatter:
        out = [formatter.format(**dct) for dct in log_dicts]
    else:
        out = log_dicts
        
    return out

def tree_to_json(filename, symbol):
    
    treedict = symbol.to_dict()
    treerepr = symbol.repr_compact()
    treetokenrepr = symbol.repr_compact(tokens=True)
    treelen = len(symbol)
    nsymbols = len(symbol._symbols)
    fulldict = {
        "repr_compact":treerepr,
        "repr_tokens":treetokenrepr,
        "tree_len":treelen,
        "num_symbols":nsymbols,
        "root_symbol":treedict,
    }
    
    fn = Path(filename)

    full = Path("./data/") / fn.with_suffix(".json")
    
    with open(str(full), 'w') as f:
        json.dump(fulldict, f, indent=3)

def test_zipper(root_token):
    
    seq = "0101011010100101011101101001010101010"
    def fetch_func(symbol, zipper):
        """
        expand branches on 1, return branch token on 0 
        """
        return symbol._token_index
    
    # zp = SymbolZipper.leaf_walker(root_token, fetch_func=fetch_func)
    zp = SymbolZipper.leaf_collector(root_token)
    
    print("leaf_collector")
    final_state = zp.run()
    print(repr(final_state))
    print(final_state.long_repr())
    
    print("confused guy")
        
    # zp = SymbolZipper.create(root_token, move_func = move_func)
    halt_func = lambda s,z:len(z.state.path) > 50
    zp = SymbolZipper.random_walker(root_token, halt_func=halt_func)
    
    final_state = zp.run()
    print(repr(final_state))
    print(final_state.long_repr())
        
    return final_state.fetched

def test_instructions():
    for inst in [wait_inst, search_inst, continue_inst, exit_inst]:
        print(inst)
        binst = inst
        while binst > 0:
            _move = binst & (2**move_bits - 1)
            move = Move(_move)
            print(_move,move.name)
            binst >>= move_bits

def test_align_sequences(node1, node2):
    
    calc = SequenceCalculator(tokenizer = tokenizer)
    
    seq1 = node1.get_tokens()
    seq2 = node2.get_tokens()
    print(f"========== comparing sequences {node1.name} and {node2.name} ==========")
    
    for a,b in zip(seq1, seq2):
        print(tokenizer.decode([a]), tokenizer.decode([b]))
    
    res = calc.compare_sequences(seq1, seq2, rough_align_threshold = 0.1)
    iters = res.aligned_pairs
    qual = res.quality
    
    
    if not iters:
        print("not similar :(")
    else:
        print(iters)
        print(qual)
        
        aligned1, aligned2 = calc.align_sequences(seq1, seq2, iters)
        
        for a, b in zip(aligned1, aligned2):
            print(tokenizer.decode([a]), tokenizer.decode([b]))
    
    print(f"========== finished comparing sequences {node1.name} and {node2.name} ==========")
    res.print_report(verbose = True)
    
    
    return res

def main():
    
    test_log_file = "./data/20250916_TerminalBridge(grant).jsonl"
    formatter = "{timestamp} {component} [{level}]: {message}"
    
    test_logs = load_log_file(test_log_file, start=0, end=5, formatter=formatter)
    
    tree = PatternTree()
    tree.set_generators(gens.get("UUID"), gens.get("DEC"), gens.get("INT"))
    
    for test_log in test_logs:
        tree.append(test_log)
        
    root = tree.root
    
    root.print_tree(info=True)
    print(root.repr_compact())
    tree_to_json("somelogs",root)
    
    # vis = test_zipper(root)
    
    
    resdict = {}
    
    for i in range(root.num_branches):
        for j in range(i, root.num_branches):
            node1 = root.children[i]
            node2 = root.children[j]
            res = test_align_sequences(node1, node2)
            resdict[node1.name, node2.name] = res.quality
            
    
    
    
        

if __name__=="__main__":
    
    main()

