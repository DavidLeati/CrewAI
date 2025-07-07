import ast
import difflib
import logging
from typing import List, Dict, Any, Tuple, Optional

class CodeReconstructor:
    def __init__(self):
        """
        Initializes the CodeReconstructor.
        """
        self.logger = logging.getLogger(__name__)
        # Configure logging if not already configured by the application
        if not self.logger.handlers:
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def reconstruct(self, reduced_code: str, reduction_map: List[Dict[str, Any]]) -> str:
        """
        Reconstructs the original code from its reduced form using the reduction map.
        It attempts to be resilient to small modifications made by an LLM to the reduced code,
        by aligning code blocks based on expected content and consuming actual lines.

        Args:
            reduced_code: The code string after reduction and potential LLM modifications.
            reduction_map: A list of dictionaries, each representing a segment of the original code.
                           Each dictionary should have:
                           - "type": "code", "comment", "docstring", "empty_line"
                           - "original_content": str (for non-code elements like comments, docstrings, empty lines)
                                                 or List[str] (for "code" type, representing original code lines)
                           - "indentation": int (original indentation level for the segment)
                           - "reduced_content_expected": Optional[List[str]] (for "code" type, the lines expected after reduction)
                           The map segments should be ordered by their original appearance in the source file.

        Returns:
            The reconstructed code string, including re-inserted comments, docstrings, and original formatting.
        """
        reconstructed_lines = []
        reduced_code_lines = reduced_code.splitlines()
        
        # current_reduced_line_idx tracks our current position in the LLM-modified reduced_code_lines.
        current_reduced_line_idx = 0 

        for i, segment in enumerate(reduction_map):
            segment_type = segment["type"]
            segment_indentation = segment["indentation"]

            if segment_type in ["comment", "docstring", "empty_line"]:
                # These elements were removed during reduction and need to be re-inserted.
                # Their content is directly taken from the original_content in the map.
                content_to_insert = segment["original_content"]
                reconstructed_lines.append(" " * segment_indentation + content_to_insert)
            elif segment_type == "code":
                expected_reduced_lines = segment.get("reduced_content_expected", [])
                
                if not expected_reduced_lines:
                    # If an empty code block was expected (e.g., a function with only a pass statement
                    # that was removed, or an empty class body), we skip it.
                    continue

                actual_code_lines_for_segment = []
                
                # --- Advanced Resilience Logic for Code Blocks ---
                
                # 1. Try to find the start of the *next* code segment in the `reduction_map`.
                # This serves as a strong anchor point for the end of the current code block.
                next_code_segment_start_idx_in_reduced_code = len(reduced_code_lines) # Default: consume till end of file
                next_expected_first_line = None

                for j in range(i + 1, len(reduction_map)):
                    if reduction_map[j]["type"] == "code":
                        next_expected_reduced_lines = reduction_map[j].get("reduced_content_expected", [])
                        if next_expected_reduced_lines:
                            next_expected_first_line = next_expected_reduced_lines[0]
                            break
                
                found_next_block_boundary = False
                if next_expected_first_line:
                    # Search for the next expected first line in the remaining `reduced_code_lines`.
                    # Use a search window to account for LLM modifications (insertions/deletions)
                    # within the current block before the next block starts.
                    search_window_start = current_reduced_line_idx
                    # Heuristic: search up to expected length + a buffer, or till end of file.
                    search_window_end = min(len(reduced_code_lines), current_reduced_line_idx + len(expected_reduced_lines) + 50) 
                    
                    best_match_idx = -1
                    best_match_ratio = 0.7 # Minimum similarity threshold for a match
                    
                    for k in range(search_window_start, search_window_end):
                        if k < len(reduced_code_lines):
                            ratio = difflib.SequenceMatcher(None, reduced_code_lines[k], next_expected_first_line).ratio()
                            if ratio > best_match_ratio:
                                best_match_ratio = ratio
                                best_match_idx = k
                                if ratio > 0.95: # Very strong match, likely the correct boundary
                                    break
                    
                    if best_match_idx != -1:
                        next_code_segment_start_idx_in_reduced_code = best_match_idx
                        found_next_block_boundary = True
                
                # 2. Consume lines for the current segment based on the determined boundary or a resilient fallback.
                if found_next_block_boundary:
                    # Scenario A: Next block's start was found.
                    # Consume lines up to that boundary and use diffing to align content.
                    candidate_actual_lines = reduced_code_lines[current_reduced_line_idx:next_code_segment_start_idx_in_reduced_code]
                    
                    matcher = difflib.SequenceMatcher(None, expected_reduced_lines, candidate_actual_lines)
                    
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == 'equal' or tag == 'replace' or tag == 'insert':
                            actual_code_lines_for_segment.extend(candidate_actual_lines[j1:j2])
                        elif tag == 'delete':
                            self.logger.debug(f"LLM deleted expected lines from segment (next block found): {expected_reduced_lines[i1:i2]}")
                    
                    current_reduced_line_idx = next_code_segment_start_idx_in_reduced_code # Advance pointer
                else:
                    # Scenario B: Could not reliably find the next block's start, or this is the last code segment.
                    # Use diffing with a lookahead window to consume lines that best align with expected content.
                    # This is the "more advanced" fallback for resilience.
                    
                    # If this is the very last segment in the map, consume all remaining lines.
                    if i == len(reduction_map) - 1:
                        while current_reduced_line_idx < len(reduced_code_lines):
                            actual_code_lines_for_segment.append(reduced_code_lines[current_reduced_line_idx])
                            current_reduced_line_idx += 1
                    else:
                        # Not the last segment, but no clear next boundary found.
                        # Take a window of actual lines to compare against expected.
                        # Heuristic: window size is expected length + a buffer for LLM insertions.
                        comparison_window_size = len(expected_reduced_lines) + 50 
                        candidate_actual_lines_for_diff = reduced_code_lines[current_reduced_line_idx : min(len(reduced_code_lines), current_reduced_line_idx + comparison_window_size)]
                        
                        matcher = difflib.SequenceMatcher(None, expected_reduced_lines, candidate_actual_lines_for_diff)
                        
                        consumed_lines_from_reduced_code_in_window = 0
                        
                        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                            if tag == 'equal' or tag == 'replace' or tag == 'insert':
                                actual_code_lines_for_segment.extend(candidate_actual_lines_for_diff[j1:j2])
                                consumed_lines_from_reduced_code_in_window = max(consumed_lines_from_reduced_code_in_window, j2)
                            elif tag == 'delete':
                                self.logger.debug(f"LLM deleted expected lines from segment (no clear next boundary): {expected_reduced_lines[i1:i2]}")
                        
                        current_reduced_line_idx += consumed_lines_from_reduced_code_in_window
                        
                        # Optional: Log a warning if the consumed lines are significantly different from expected
                        # in this fallback scenario, indicating potential major LLM deviation.
                        if len(expected_reduced_lines) > 0 and \
                           (len(actual_code_lines_for_segment) < len(expected_reduced_lines) * 0.5 or \
                            len(actual_code_lines_for_segment) > len(expected_reduced_lines) * 2.0):
                            self.logger.warning(
                                f"Reconstruction of code segment (no clear next boundary) resulted in "
                                f"significant line count deviation. Expected: {len(expected_reduced_lines)}, "
                                f"Actual: {len(actual_code_lines_for_segment)}. "
                                f"Segment map index: {i}"
                            )

                # Apply the original indentation to each line of the consumed code segment.
                for line in actual_code_lines_for_segment:
                    reconstructed_lines.append(" " * segment_indentation + line)
            else:
                # Handle unknown segment types gracefully by logging a warning.
                # This indicates an issue with the reduction_map itself.
                self.logger.warning(f"Unknown segment type '{segment_type}' encountered in reduction map. Skipping segment.")

        # After processing all segments from the map, append any remaining lines
        # from `reduced_code_lines`. This catches any code added by the LLM
        # at the very end of the file that wasn't part of a mapped segment.
        while current_reduced_line_idx < len(reduced_code_lines):
            reconstructed_lines.append(reduced_code_lines[current_reduced_line_idx])
            current_reduced_line_idx += 1

        return "\n".join(reconstructed_lines)