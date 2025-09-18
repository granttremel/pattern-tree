
from typing import List

from pattern_tree.pattern_tree import encode_string, decode_string, integer_generator, decimal_generator, Symbol, SymbolGenerator

import json
import re

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

def test_encode_decode(input_str):
    
    nchars = len(input_str)
    print(f"original string ({nchars} chars):")
    print(input_str)
    
    encoded = encode_string(input_str)
    ntokens = len(encoded)
    print(f"after encoding, ({ntokens} tks):")
    print(encoded)
    modified = [enc + 1 for enc in encoded]
    decoded = decode_string(modified)
    new_nchars = len(decoded)
    print(f"after modification ({new_nchars} chars):")
    print(decoded)

def test_number_extraction(input_str):
    
    decimals = decimal_generator.detect(input_str)
    ints = integer_generator.detect(input_str)
    
    return decimals, ints

def test_number_substitution(input_str):
    new_str = decimal_generator.substitute(input_str)
    final_str = integer_generator.substitute(new_str)
    return final_str

def test_get_number_symbols(input_str:str):
    
    gens = [decimal_generator, integer_generator]
    out_symbols = []
    out_strs=[input_str]
    
    for gen in gens:
        gensyms = gen.extract(input_str)
        
        for gensym in gensyms:
            sym = gen.to_symbol(gensym)
            out_symbols.append(sym)
            
            input_str_split = gen.split(input_str, gensym)
            print(input_str_split)
            # input_str = input_str.replace(str(sym),sym.name)
        
            
    return out_symbols, input_str

def test_extract_split(gen:SymbolGenerator, symbol:str):
    
    working = [symbol]
    imax = len(working)
    
    gensyms = gen.extract(symbol)
    
    i=0
    
    for sym in  gensyms:
        current = working[i]
        if isinstance(current, Symbol):
            continue
        
        split = gen.split(current, sym)
        working = split
        imax = len(working)
        i += 1
        
        if i >= imax:
            break
    
    return gensyms, working

def multiple_extract_split(gens, input_symbols):
    
    if isinstance(input_symbols, str):
        input_symbols = [input_symbols]
    
    all_syms = []
    
    for gen in gens:
        _input_symbols = []
        for input_symbol in input_symbols:
            syms, new_input_symbols = test_extract_split(gen, input_symbols)
            _input_symbols.extend(new_input_symbols)
        input_symbols = _input_symbols
        all_syms.extend(syms)
        
    return all_syms, input_symbols

def test_extract_divide(root_symbol, gen):
    syms = gen.extract(root_symbol)
    print(syms)
    for sym in syms:
        # root_symbol.divide(sym, at_root=False)
        root_symbol.divide(sym, at_root=True)
        # root_symbol // sym
    
    return syms

def test_symbol_names():
    
    for _str in "asckhnofhqornc093qdq3br4g4v9pqwecjh0etjn-rfj4w098jrpbne08fj4-fgvm":
        new_sym = Symbol.get_symbol(_str)
        print(repr(new_sym))
        
    pass

def main():
    
    test_str = "hello this is an example of natural language"
    
    # root_symbol, syms = test_extract_divide(test_str, decimal_generator)
    
    # print(root_symbol)
    # print(syms)
    
    # test_symbol_names()
    
    Symbol._clear()
    
    
    test_log_file = "./data/20250916_TerminalBridge(grant).jsonl"

    formatter = "{timestamp} {component} [{level}]: {message}"
    
    test_logs = load_log_file(test_log_file, start=1, end=2, formatter=formatter)
    
    # test_logs = ["this is a better test string 2025 and i hope it works 40.91921 okay here goes"]
    
    for test_str in test_logs:
        root_symbol = Symbol.get_symbol(test_str)
        syms = test_extract_divide(root_symbol, decimal_generator)
        # print(repr(root_symbol))
        syms = test_extract_divide(root_symbol, integer_generator)
        print(repr(root_symbol))
        # print('hi')


        # Symbol._clear()
        
        
if __name__=="__main__":
    main()

