# Author Ordering Fix for Ablation Studies

## Problem Description

The original implementation had a critical issue with author ordering that made ablation studies unreliable. When performing experiments like "drop first author" or "drop last author", the system was not guaranteed to drop the actual first or last author from the publication because:

1. **Edge Index Order**: The `edge_index` in PyTorch Geometric doesn't preserve the original publication order of authors
2. **Unordered Author Lists**: The `collect_neighbours` method returned authors in the order they appeared in the edge index, not publication order
3. **Inconsistent Ablation**: Ablation functions were applied to potentially unordered author lists
4. **Temporal Sequence Issues**: The `predict_teams` and `evaluate_team` methods didn't properly handle author evolution over time

## Solution Implemented

### 1. Modified Dataset Builder (`utils/dataset_builder.py`)

- **Added Author Order Storage**: The `build_snapshot` function now stores author order information for each paper
- **Preserved Publication Order**: Authors are stored with their original position in the publication (first author = 0, second author = 1, etc.)
- **Backward Compatibility**: The changes are backward compatible with existing code

```python
# New code in build_snapshot function:
author_order_info = {}  # Store author order for each paper

for pid in paper_ids:
    p = PAPER_LIDX_OF[pid]
    node = paper_json[pid]
    
    # Store author order information
    author_order_info[p] = []
    
    for i, aid in enumerate(node['neighbors']['author']):
        a_src.append(AUTH_LIDX_OF[aid])
        a_dst.append(p)
        author_order_info[p].append((AUTH_LIDX_OF[aid], i))  # (author_idx, order)

# Store author order information in the graph
data['paper'].author_order = author_order_info
```

### 2. Updated Imputer (`models/imputer.py`)

- **Ordered Author Collection**: The `collect_neighbours` method now returns authors in their original publication order
- **Fallback Support**: If author order information is not available, it falls back to the original edge index method
- **Consistent Interface**: The method signature and return format remain the same

```python
# Updated collect_neighbours method:
if hasattr(data['paper'], 'author_order') and paper_id in data['paper'].author_order:
    # Get authors in their original publication order
    author_order_list = data['paper'].author_order[paper_id]
    # Sort by order (second element of tuple) and extract author indices
    sorted_authors = sorted(author_order_list, key=lambda x: x[1])
    author_indices = torch.tensor([author_idx for author_idx, _ in sorted_authors], 
                                dtype=torch.long, device=device)
    neighbours['author'] = author_indices
else:
    # Fallback to edge index method (original behavior)
    # ... original code ...
```

### 3. Fixed Ablation Application (`models/full_model.py`)

- **Removed Double Application**: Fixed the issue where ablation functions were applied twice
- **Cleaner Logic**: Ablation is now applied only once in the `evaluate_team` method

### 4. Fixed Temporal Sequence Building (`models/full_model.py`)

- **Consistent Sequence Construction**: Both `predict_team` and `predict_teams` now use the same approach as the forward method
- **Proper Author Evolution**: The sequence correctly handles cases where authors may not exist in earlier years
- **Imputer Integration**: Uses the `imputer` method with `predefined_neigh` parameter for consistent behavior

```python
# Fixed predict_teams method:
seq_steps = []
for k in range(history + 1):  # 0, 1, 2, 3, 4, 5
    yr = current_year_idx - k
    
    # Build sequence step for all teams at once
    seq_k = torch.stack([
        self.imputer(
            None,  # paper_id not needed
            yr,
            snapshots,
            embeddings,
            predefined_neigh={'author': self._get_author_indices_for_year(teams[i][0], yr, snapshots)},
            topic_vec=teams[i][1]
        )
        for i in range(N)
    ], dim=0)  # [N, H]
    
    seq_steps.append(seq_k)
```

## Key Improvements

### Temporal Sequence Consistency

The most important fix addresses the temporal sequence building issue:

**Before (Problematic):**
- `predict_teams` tried to manually build sequences using raw author IDs
- Authors that didn't exist in earlier years caused errors
- Inconsistent with the forward method's approach

**After (Fixed):**
- Uses the same `imputer` method as the forward method
- Properly handles missing authors in earlier years
- Consistent sequence construction across all methods

### Author Evolution Handling

The new implementation correctly handles author evolution over time:

```python
def _get_author_indices_for_year(self, author_ids: list[str], year: int, snapshots: list) -> torch.Tensor:
    """Helper method to get author indices for a specific year."""
    device = next(self.parameters()).device
    
    if not author_ids:
        return torch.empty(0, dtype=torch.long, device=device)
    
    # Get raw author IDs for the specified year
    raw_ids = snapshots[year]['author'].raw_ids
    raw2row = {aid: i for i, aid in enumerate(raw_ids)}
    
    # Find which authors exist in this year
    rows = [raw2row[a] for a in author_ids if a in raw2row]
    
    if not rows:
        return torch.empty(0, dtype=torch.long, device=device)
    
    return torch.tensor(rows, dtype=torch.long, device=device)
```

This ensures that:
- Authors who don't exist in a particular year are properly handled
- The sequence maintains consistency with the training approach
- No errors occur when authors are missing from earlier snapshots

## Usage

### Running Ablation Studies

The ablation functions now work correctly with ordered authors and proper temporal sequences:

```python
# Example usage in train.py
if args.inference_time_author_dropping=='drop_first':
    drop_fn = drop_first
elif args.inference_time_author_dropping=='drop_last':
    drop_fn = drop_last
elif args.inference_time_author_dropping=='keep_first':
    drop_fn = keep_first
elif args.inference_time_author_dropping=='keep_last':
    drop_fn = keep_last
elif args.inference_time_author_dropping=='drop_first_and_last':
    drop_fn = drop_first_and_last
else:
    drop_fn = None

# The ablation function will now correctly operate on ordered authors
# and the temporal sequence will be built consistently
male, rmsle, mape, y_true, y_predict = model.evaluate_team(
    snapshots,
    test_years,
    start_year=train_years[0],
    return_raw=True,
    author_drop_fn=drop_fn
)
```

### Available Ablation Functions

- `drop_first`: Removes the first author (index 0)
- `drop_last`: Removes the last author (index -1)
- `drop_first_and_last`: Removes both first and last authors
- `keep_first`: Keeps only the first author
- `keep_last`: Keeps only the last author
- `drop_all`: Removes all authors
- `drop_none`: Keeps all authors (no ablation)

## Testing

Two test scripts have been created to verify the implementation:

### 1. Author Order Test (`test_author_order.py`)

```bash
cd GNNteamingEvaluator/TeamingEvaluator
python test_author_order.py
```

This script tests:
- Author order information storage
- `collect_neighbours` method correctness
- Ablation function behavior

### 2. Temporal Sequence Test (`test_temporal_sequence.py`)

```bash
cd GNNteamingEvaluator/TeamingEvaluator
python test_temporal_sequence.py
```

This script tests:
- Temporal sequence building across multiple years
- Author evolution over time
- Model helper methods

## Verification

To verify that the fix is working:

1. **Check Author Order**: The test script will show that authors are returned in publication order
2. **Temporal Consistency**: Verify that sequences are built consistently across all methods
3. **Ablation Results**: When you run ablation studies, the results should now be meaningful:
   - "Drop first author" will actually drop the first author
   - "Drop last author" will actually drop the last author
   - Citation predictions will reflect the impact of removing the correct authors
4. **No Errors**: The system should handle missing authors gracefully without errors

## Important Notes

1. **Data Regeneration**: You may need to regenerate your snapshots to include the author order information
2. **Backward Compatibility**: Existing code will continue to work, but without author order preservation
3. **Performance**: The changes add minimal overhead and maintain the same computational complexity
4. **Temporal Consistency**: All methods now use the same sequence building approach as the forward method

## Example Results

With the fix, your ablation studies should now produce meaningful results:

- **Original Team**: [Author1, Author2, Author3, Author4] → Predicted citations: 50
- **Drop First Author**: [Author2, Author3, Author4] → Predicted citations: 35
- **Drop Last Author**: [Author1, Author2, Author3] → Predicted citations: 42
- **Keep First Only**: [Author1] → Predicted citations: 20

The temporal sequence will correctly handle cases where:
- Author1 and Author2 existed in year t-3
- Only Author1 existed in year t-4
- None of the authors existed in year t-5

This allows you to properly analyze the contribution of different author positions to paper impact while maintaining temporal consistency. 