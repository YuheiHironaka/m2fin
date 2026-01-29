import re

def process_author_field(val):
    if ' and ' in val or '，' in val:
        # Split by the first ' and ' or '，'
        first = re.split(r'\s+and\s+|，', val)[0].strip()
        return first + " and others"
    return val

def fix_bib(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    output = []
    pos = 0
    # Use re.finditer to avoid string slicing in loop for start index
    pattern = re.compile(r'\bauthor\s*=\s*', re.IGNORECASE)
    
    for match in pattern.finditer(content):
        # Add everything before the match
        output.append(content[pos:match.start()])
        
        field_pos = match.end()
        if field_pos >= len(content):
            output.append(content[match.start():])
            pos = len(content)
            break
            
        delim = content[field_pos]
        if delim == '{':
            end_delim = '}'
            balance = 1
            author_start = field_pos + 1
            current = author_start
            while balance > 0 and current < len(content):
                if content[current] == '{':
                    balance += 1
                elif content[current] == '}':
                    balance -= 1
                current += 1
            author_val = content[author_start:current-1]
            new_val = process_author_field(author_val)
            output.append(f'author = {{{new_val}}}')
            pos = current
        elif delim == '"':
            author_start = field_pos + 1
            author_end = content.find('"', author_start)
            if author_end == -1:
                output.append(content[match.start():])
                pos = len(content)
                break
            author_val = content[author_start:author_end]
            new_val = process_author_field(author_val)
            output.append(f'author = "{new_val}"')
            pos = author_end + 1
        else:
            # Fallback if delimiter is something else or missing
            output.append(content[match.start():match.end()])
            pos = match.end()
            
    # Add any remaining content
    if pos < len(content):
        output.append(content[pos:])
            
    return "".join(output)

fixed = fix_bib('books.bib')
with open('books_pre.bib', 'w', encoding='utf-8') as f:
    f.write(fixed)
