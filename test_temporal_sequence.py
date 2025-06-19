#!/usr/bin/env python3
"""
Test script to verify temporal sequence building with proper author handling.
This script helps debug the temporal sequence generation in predict_teams and predict_team.
"""

import torch
import json
import gzip
from utils.dataset_builder import build_snapshot
from models.imputer import WeightedImputer
from models.full_model import ImpactModel

def test_temporal_sequence():
    """Test that temporal sequences are built correctly with author evolution."""
    
    print("Testing Temporal Sequence Building")
    print("=" * 50)
    
    # Build snapshots for multiple years to test temporal evolution
    years_to_test = [2008, 2009, 2010, 2011, 2012, 2013]
    snapshots = []
    
    print("Building snapshots for years:", years_to_test)
    for year in years_to_test:
        try:
            snapshot = build_snapshot(up_to_year=year)
            snapshots.append(snapshot)
            print(f"  ✓ Built snapshot for year {year} with {len(snapshot['author'].raw_ids)} authors")
        except Exception as e:
            print(f"  ✗ Failed to build snapshot for year {year}: {e}")
            return
    
    if len(snapshots) < 2:
        print("✗ Need at least 2 snapshots to test temporal sequence")
        return
    
    # Test the imputer's temporal behavior
    print("\nTesting Imputer Temporal Behavior:")
    print("-" * 40)
    
    imputer = WeightedImputer(("author", "venue", "paper"), 768)
    
    # Find a paper that exists in the latest snapshot
    latest_snapshot = snapshots[-1]
    if hasattr(latest_snapshot['paper'], 'author_order'):
        test_papers = list(latest_snapshot['paper'].author_order.keys())[:3]
        
        for paper_idx in test_papers:
            print(f"\nPaper {paper_idx}:")
            
            # Get current year authors
            current_authors = latest_snapshot['paper'].author_order[paper_idx]
            current_author_ids = [latest_snapshot['author'].raw_ids[author_idx] for author_idx, _ in current_authors]
            print(f"  Current year authors: {current_author_ids}")
            
            # Test how these authors appear in previous years
            for i, snapshot in enumerate(snapshots[:-1]):  # Skip the latest
                year = years_to_test[i]
                print(f"  Year {year}:")
                
                # Check which authors exist in this year
                available_authors = []
                for author_id in current_author_ids:
                    if author_id in snapshot['author'].raw_ids:
                        available_authors.append(author_id)
                
                print(f"    Available authors: {available_authors}")
                
                # Test imputer with these authors
                author_indices = []
                for author_id in available_authors:
                    author_idx = snapshot['author'].raw_ids.index(author_id)
                    author_indices.append(author_idx)
                
                if author_indices:
                    author_tensor = torch.tensor(author_indices, dtype=torch.long)
                    print(f"    Author indices: {author_indices}")
                    
                    # Test imputer call
                    try:
                        embedding = imputer(
                            None,
                            i,  # year index
                            snapshots,
                            [None] * len(snapshots),  # dummy embeddings
                            predefined_neigh={'author': author_tensor},
                            topic_vec=torch.zeros(768)
                        )
                        print(f"    ✓ Imputer succeeded, embedding shape: {embedding.shape}")
                    except Exception as e:
                        print(f"    ✗ Imputer failed: {e}")
                else:
                    print(f"    No authors available in year {year}")

def test_model_temporal_sequence():
    """Test the model's temporal sequence building methods."""
    
    print("\nTesting Model Temporal Sequence Methods:")
    print("-" * 50)
    
    # This would require a trained model, so we'll just test the helper method
    print("Note: Full model testing requires a trained model")
    print("Testing helper method _get_author_indices_for_year...")
    
    # Build a simple test case
    years_to_test = [2010, 2011, 2012]
    snapshots = []
    
    for year in years_to_test:
        try:
            snapshot = build_snapshot(up_to_year=year)
            snapshots.append(snapshot)
        except Exception as e:
            print(f"Failed to build snapshot for year {year}: {e}")
            return
    
    # Create a mock model to test the helper method
    class MockModel:
        def __init__(self):
            self.device = torch.device('cpu')
        
        def _get_author_indices_for_year(self, author_ids, year, snapshots):
            device = self.device
            
            if not author_ids:
                return torch.empty(0, dtype=torch.long, device=device)
            
            raw_ids = snapshots[year]['author'].raw_ids
            raw2row = {aid: i for i, aid in enumerate(raw_ids)}
            
            rows = [raw2row[a] for a in author_ids if a in raw2row]
            
            if not rows:
                return torch.empty(0, dtype=torch.long, device=device)
            
            return torch.tensor(rows, dtype=torch.long, device=device)
    
    mock_model = MockModel()
    
    # Test with some sample author IDs
    if len(snapshots) > 0 and len(snapshots[0]['author'].raw_ids) > 0:
        test_author_ids = snapshots[0]['author'].raw_ids[:3]  # First 3 authors
        print(f"Test author IDs: {test_author_ids}")
        
        for i, snapshot in enumerate(snapshots):
            year = i
            indices = mock_model._get_author_indices_for_year(test_author_ids, year, snapshots)
            print(f"  Year {year}: {indices.tolist()}")
            
            # Verify the indices are correct
            if indices.numel() > 0:
                for idx in indices:
                    if idx < len(snapshot['author'].raw_ids):
                        author_id = snapshot['author'].raw_ids[idx]
                        if author_id in test_author_ids:
                            print(f"    ✓ Author {author_id} found at index {idx}")
                        else:
                            print(f"    ✗ Unexpected author {author_id} at index {idx}")
                    else:
                        print(f"    ✗ Index {idx} out of bounds")

def test_author_evolution():
    """Test how author lists evolve over time."""
    
    print("\nTesting Author Evolution Over Time:")
    print("-" * 40)
    
    # Build snapshots for consecutive years
    years = [2008, 2009, 2010, 2011, 2012, 2013]
    snapshots = []
    
    for year in years:
        try:
            snapshot = build_snapshot(up_to_year=year)
            snapshots.append(snapshot)
        except Exception as e:
            print(f"Failed to build snapshot for year {year}: {e}")
            return
    
    print(f"Built {len(snapshots)} snapshots")
    
    # Track author growth
    author_counts = []
    for i, snapshot in enumerate(snapshots):
        count = len(snapshot['author'].raw_ids)
        author_counts.append(count)
        print(f"  Year {years[i]}: {count} authors")
    
    # Check if authors are being added over time
    if len(author_counts) > 1:
        is_growing = all(author_counts[i] <= author_counts[i+1] for i in range(len(author_counts)-1))
        if is_growing:
            print("✓ Author count is growing over time (as expected)")
        else:
            print("✗ Author count is not consistently growing")
    
    # Test author persistence
    if len(snapshots) >= 2:
        first_year_authors = set(snapshots[0]['author'].raw_ids)
        last_year_authors = set(snapshots[-1]['author'].raw_ids)
        
        # Authors from first year should exist in last year
        persistent_authors = first_year_authors.intersection(last_year_authors)
        print(f"Authors from {years[0]} that persist to {years[-1]}: {len(persistent_authors)}/{len(first_year_authors)}")
        
        if len(persistent_authors) == len(first_year_authors):
            print("✓ All early authors persist (expected behavior)")
        else:
            print("⚠ Some early authors don't persist (this might be normal)")

if __name__ == "__main__":
    print("Testing Temporal Sequence Generation")
    print("=" * 60)
    
    try:
        test_temporal_sequence()
        test_model_temporal_sequence()
        test_author_evolution()
        print("\n✓ All temporal sequence tests completed!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 