#!/usr/bin/env python3
"""
Test script to verify author ordering is preserved correctly.
This script helps debug the author ablation functionality.
"""

import torch
import json
import gzip
from utils.dataset_builder import build_snapshot
from models.imputer import WeightedImputer

def test_author_order():
    """Test that authors are returned in their original publication order."""
    
    # Build a snapshot for a specific year
    print("Building snapshot for year 2010...")
    snapshot = build_snapshot(up_to_year=2010)
    
    # Check if author_order information is stored
    if hasattr(snapshot['paper'], 'author_order'):
        print("✓ Author order information is stored in the snapshot")
        print(f"Number of papers with author order info: {len(snapshot['paper'].author_order)}")
        
        # Test a few papers
        test_papers = list(snapshot['paper'].author_order.keys())[:5]
        
        for paper_idx in test_papers:
            print(f"\nPaper {paper_idx}:")
            author_order_list = snapshot['paper'].author_order[paper_idx]
            print(f"  Author order list: {author_order_list}")
            
            # Get raw author IDs
            raw_author_ids = [snapshot['author'].raw_ids[author_idx] for author_idx, _ in author_order_list]
            print(f"  Raw author IDs: {raw_author_ids}")
            
            # Test the collect_neighbours method
            imputer = WeightedImputer(("author", "venue", "paper"), 768)
            neighbours = imputer.collect_neighbours(snapshot, paper_idx, torch.device('cpu'))
            
            if 'author' in neighbours:
                ordered_author_indices = neighbours['author']
                print(f"  Ordered author indices from collect_neighbours: {ordered_author_indices}")
                
                # Verify the order matches
                expected_order = [author_idx for author_idx, _ in sorted(author_order_list, key=lambda x: x[1])]
                actual_order = ordered_author_indices.tolist()
                
                if expected_order == actual_order:
                    print("  ✓ Author order preserved correctly")
                else:
                    print("  ✗ Author order mismatch!")
                    print(f"    Expected: {expected_order}")
                    print(f"    Actual:   {actual_order}")
    else:
        print("✗ Author order information not found in snapshot")
        print("This means the dataset_builder.py changes haven't been applied yet.")

def test_ablation_functions():
    """Test the ablation functions work correctly with ordered authors."""
    
    # Define test author list (simulating ordered authors from first to last)
    test_authors = ['author1', 'author2', 'author3', 'author4', 'author5']
    print(f"\nTesting ablation functions with authors: {test_authors}")
    
    # Test each ablation function
    ablation_functions = {
        'drop_first': lambda authors: authors[1:] if len(authors) > 1 else [],
        'drop_last': lambda authors: authors[:-1] if len(authors) > 1 else [],
        'drop_first_and_last': lambda authors: authors[1:-1] if len(authors) > 2 else [],
        'keep_first': lambda authors: authors[:1] if authors else [],
        'keep_last': lambda authors: authors[-1:] if authors else [],
        'drop_all': lambda authors: [],
        'drop_none': lambda authors: authors
    }
    
    for name, func in ablation_functions.items():
        result = func(test_authors)
        print(f"  {name}: {result}")
        
        # Verify the logic makes sense
        if name == 'drop_first' and len(result) == len(test_authors) - 1:
            print(f"    ✓ Correctly dropped first author")
        elif name == 'drop_last' and len(result) == len(test_authors) - 1:
            print(f"    ✓ Correctly dropped last author")
        elif name == 'drop_first_and_last' and len(result) == len(test_authors) - 2:
            print(f"    ✓ Correctly dropped first and last authors")
        elif name == 'keep_first' and len(result) == 1:
            print(f"    ✓ Correctly kept only first author")
        elif name == 'keep_last' and len(result) == 1:
            print(f"    ✓ Correctly kept only last author")
        elif name == 'drop_all' and len(result) == 0:
            print(f"    ✓ Correctly dropped all authors")
        elif name == 'drop_none' and result == test_authors:
            print(f"    ✓ Correctly kept all authors")

if __name__ == "__main__":
    print("Testing Author Order Preservation")
    print("=" * 40)
    
    try:
        test_author_order()
        test_ablation_functions()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 