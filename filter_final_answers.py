#!/usr/bin/env python3
"""
Filter loc_outputs.jsonl to keep only files from final ASSISTANT responses.
"""

import json
import re
from typing import List, Tuple

def parse_structured_response(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse structured FILES/FUNCTIONS format from assistant text."""
    found_files = []
    found_modules = []
    found_entities = []
    
    lines = text.split('\n')
    in_files = False
    in_functions = False
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('FILES:'):
            in_files, in_functions = True, False
            continue
        if line.upper().startswith('FUNCTIONS:'):
            in_files, in_functions = False, True
            continue
        
        if line.startswith('- ') or line.startswith('* '):
            line = line[2:].strip()
        line = line.strip('`')
        
        if in_files and line and (line.endswith('.py') or '/' in line):
            if line not in found_files:
                found_files.append(line)
        
        if in_functions and line and ':' in line:
            parts = line.split(':', 1)
            file_path, func_part = parts[0], parts[1]
            entity = f"{file_path}:{func_part}"
            
            if entity not in found_entities:
                found_entities.append(entity)
            if file_path not in found_files:
                found_files.append(file_path)
            
            # Extract module
            if '.' in func_part:
                class_name = func_part.split('.')[0]
                module_loc = f"{file_path}:{class_name}"
            else:
                module_loc = entity
            if module_loc not in found_modules:
                found_modules.append(module_loc)
    
    return found_files, found_modules, found_entities


def extract_final_assistant_response(raw_response: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract only the final ASSISTANT message from raw response."""
    found_files = []
    found_modules = []
    found_entities = []
    
    # Parse JSONL events
    for line in raw_response.strip().split('\n'):
        try:
            event = json.loads(line)
            event_type = event.get('type', '')
            
            if event_type == 'item.completed':
                item = event.get('item', {})
                item_type = item.get('type', item.get('item_type', ''))
                
                # Only parse ASSISTANT messages (not commands)
                if item_type in ('assistant_message', 'agent_message'):
                    text = item.get('text', '')
                    
                    # Parse structured output
                    parsed_files, parsed_modules, parsed_entities = parse_structured_response(text)
                    
                    # Replace (not append) - we want only the LAST assistant message
                    found_files = parsed_files
                    found_modules = parsed_modules
                    found_entities = parsed_entities
                    
        except json.JSONDecodeError:
            continue
    
    return found_files, found_modules, found_entities


def main():
    input_file = "results/codex_agent/loc_outputs.jsonl"
    output_file = "results/codex_agent/loc_outputs_filtered.jsonl"
    
    filtered_results = []
    
    with open(input_file) as f:
        for line in f:
            result = json.loads(line)
            
            # Store original counts BEFORE modifying
            original_file_count = len(result.get('found_files', []))
            original_module_count = len(result.get('found_modules', []))
            original_entity_count = len(result.get('found_entities', []))
            
            # Only re-parse successful results
            if result.get('status') == 'FINISHED' and result.get('raw_response'):
                raw_response = result['raw_response']
                
                # Extract only final assistant response
                final_files, final_modules, final_entities = extract_final_assistant_response(raw_response)
                
                # Update result with filtered data
                result['found_files'] = final_files
                result['found_modules'] = final_modules
                result['found_entities'] = final_entities
                
                # Store counts for comparison
                result['_original_file_count'] = original_file_count
                result['_filtered_file_count'] = len(final_files)
                result['_original_module_count'] = original_module_count
                result['_filtered_module_count'] = len(final_modules)
                result['_original_entity_count'] = original_entity_count
                result['_filtered_entity_count'] = len(final_entities)
            
            filtered_results.append(result)
    
    # Save filtered results
    with open(output_file, 'w') as f:
        for result in filtered_results:
            f.write(json.dumps(result) + '\n')
    
    # Summary
    print(f"Filtered {len(filtered_results)} results")
    print(f"Output: {output_file}")
    
    # Stats
    total_original = 0
    total_filtered = 0
    finished_count = 0
    
    for r in filtered_results:
        if r.get('status') == 'FINISHED':
            finished_count += 1
            total_original += r.get('_original_file_count', 0)
            total_filtered += r.get('_filtered_file_count', 0)
    
    print(f"Finished instances: {finished_count}/{len(filtered_results)}")
    print(f"Total files before filtering: {total_original}")
    print(f"Total files after filtering: {total_filtered}")
    if total_original > 0:
        reduction_pct = 100 * (1 - total_filtered / total_original)
        print(f"Reduction: {total_original - total_filtered} files ({reduction_pct:.1f}%)")
    else:
        print("No files to filter")
    
    # Show some examples
    print("\nExample comparisons:")
    count = 0
    for r in filtered_results:
        if r.get('_original_file_count', 0) != r.get('_filtered_file_count', 0):
            print(f"  {r['instance_id']}: {r.get('_original_file_count')} -> {r.get('_filtered_file_count')} files")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()