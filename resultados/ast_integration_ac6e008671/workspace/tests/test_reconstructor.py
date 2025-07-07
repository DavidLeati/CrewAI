import unittest
import textwrap
import re
import logging

# Configure logging for the mock to signal discrepancies
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MockReducer:
    """
    A mock Reducer that simulates reducing Python code by removing comments
    and normalizing whitespace, while capturing metadata necessary for reconstruction.
    """
    def reduce(self, code: str) -> tuple[str, list]:
        reduced_lines = []
        metadata = []
        
        # Use splitlines(keepends=True) to preserve original line endings
        lines = code.splitlines(keepends=True)
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            is_comment_or_empty = stripped_line == '' or stripped_line.startswith('#')
            
            if not is_comment_or_empty:
                # For the reduced code, we keep the stripped version
                reduced_lines.append(stripped_line)
            
            # Store metadata for each original line
            metadata.append({
                'original_line_content': line, # Full original line including ending
                'is_code_line': not is_comment_or_empty,
                'original_indent': len(line) - len(line.lstrip())
            })
        
        # The reduced code itself is joined by standard newlines, as it's a "core" representation.
        return "\n".join(reduced_lines), metadata

class MockReconstructor:
    """
    A mock Reconstructor that rebuilds code from a reduced version and original metadata.
    It aims for byte-by-byte accuracy for structural elements (indentation, comments, empty lines)
    and integrates potentially modified code content from the reduced input.
    
    This version is enhanced to handle LLM-added lines by attempting to align
    the reduced code with the original metadata. It signals discrepancies
    (added or potentially removed lines).
    """
    def __init__(self, original_metadata: list):
        self.original_metadata = original_metadata
        self.discrepancies = [] # To store info about lines added/removed by LLM

    def reconstruct(self, reduced_code: str) -> str:
        reconstructed_lines = []
        reduced_lines_list = reduced_code.splitlines()
        reduced_idx = 0 # Pointer for reduced_lines_list

        for meta_idx, meta in enumerate(self.original_metadata):
            original_line_ending = meta['original_line_content'][len(meta['original_line_content'].rstrip('\r\n')):]
            indent = ' ' * meta['original_indent']

            if not meta['is_code_line']:
                # If it was a comment or empty line, restore it exactly as it was
                reconstructed_lines.append(meta['original_line_content'])
            else:
                # This was an original code line. We need to find its corresponding
                # line in the reduced_code.
                
                original_stripped_content = meta['original_line_content'].strip()
                
                found_match_for_current_meta = False
                
                # Look for matching reduced lines that might correspond to this original line
                # or are new lines inserted before the next original line.
                while reduced_idx < len(reduced_lines_list):
                    current_reduced_line_candidate = reduced_lines_list[reduced_idx]
                    
                    # Heuristic for matching:
                    # 1. Exact match.
                    # 2. If the first significant token (before '(' or '=') is the same.
                    # This allows for modifications to the line content while preserving structure.
                    is_match = False
                    if original_stripped_content == current_reduced_line_candidate:
                        is_match = True
                    elif original_stripped_content and current_reduced_line_candidate:
                        # Improved prefix matching: consider words, not just until first paren/equals
                        original_tokens = re.findall(r'\b\w+\b', original_stripped_content)
                        candidate_tokens = re.findall(r'\b\w+\b', current_reduced_line_candidate)
                        
                        if original_tokens and candidate_tokens and original_tokens[0] == candidate_tokens[0]:
                            is_match = True
                        # Also, if one is a substring of the other (e.g., 'foo' vs 'foo.bar()')
                        elif original_stripped_content in current_reduced_line_candidate or \
                             current_reduced_line_candidate in original_stripped_content:
                            is_match = True

                    if is_match:
                        # Found the line corresponding to the current metadata entry
                        reconstructed_lines.append(indent + current_reduced_line_candidate + original_line_ending)
                        reduced_idx += 1
                        found_match_for_current_meta = True
                        break # Move to the next metadata entry
                    else:
                        # This line does not seem to correspond to the current original line.
                        # Assume it's an LLM-added line.
                        # Append it with the current meta's indentation.
                        reconstructed_lines.append(indent + current_reduced_line_candidate + '\n') # Add with current indentation
                        self.discrepancies.append(f"Added: {current_reduced_line_candidate}")
                        reduced_idx += 1
                
                if not found_match_for_current_meta:
                    # If we exhausted reduced_lines_list without finding a match for the current original line,
                    # it means the original line was removed by the LLM.
                    # logging.warning(f"MockReconstructor: Original line '{original_stripped_content}' not found in reduced code (likely removed).")
                    self.discrepancies.append(f"Removed: {original_stripped_content}")
                    # For a mock, we can re-add the original content to show what was removed
                    # For a real reconstructor, this might be a gap or an error.
                    reconstructed_lines.append(meta['original_line_content']) 

        # After processing all original metadata, append any remaining lines from reduced_code.
        # These are lines added by LLM at the very end.
        while reduced_idx < len(reduced_lines_list):
            remaining_line = reduced_lines_list[reduced_idx]
            # For remaining lines, use indentation of the last processed code line, or default to 0.
            # A simple approach is to append with a newline and signal.
            # If there was a previous code line, try to match its indentation.
            last_code_indent = 0
            for i in reversed(range(len(self.original_metadata))):
                if self.original_metadata[i]['is_code_line']:
                    last_code_indent = self.original_metadata[i]['original_indent']
                    break
            
            reconstructed_lines.append(' ' * last_code_indent + remaining_line + '\n')
            self.discrepancies.append(f"Added (end): {remaining_line}")
            reduced_idx += 1
        
        return "".join(reconstructed_lines)

class TestReconstructor(unittest.TestCase):
    """
    Unit and integration tests for the Reconstructor module.
    Focuses on exact reconstruction and resilience to minor LLM modifications.
    """

    def setUp(self):
        """Set up a new MockReducer for each test."""
        self.reducer = MockReducer()

    def test_exact_reconstruction_simple_function(self):
        """
        Tests byte-by-byte reconstruction of a simple Python function
        including comments and indentation.
        """
        original_code = textwrap.dedent("""\
            def greet(name):
                # This is a greeting function
                print(f"Hello, {name}!") # Say hello
            """)
        
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_exact_reconstruction_class_definition(self):
        """
        Tests byte-by-byte reconstruction of a Python class definition
        with methods and docstrings.
        """
        original_code = textwrap.dedent("""\
            class MyClass:
                \"\"\"
                A simple class for demonstration.
                \"\"\"
                def __init__(self, value):
                    self.value = value # Store value
                
                def get_value(self):
                    # Returns the stored value
                    return self.value
            """)
        
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_exact_reconstruction_with_various_elements(self):
        """
        Tests byte-by-byte reconstruction with a mix of imports, functions,
        comments, empty lines, and main block.
        """
        original_code = textwrap.dedent("""\
            # This is a test file
            
            import os # Import statement
            
            def calculate_sum(a, b):
                \"\"\"
                Calculates the sum of two numbers.
                \"\"\"
                result = a + b # Perform addition
                return result
            
            
            if __name__ == "__main__":
                x = 10
                y = 20
                total = calculate_sum(x, y) # Call the function
                print(f"The sum is: {total}") # Print result
            """)
        
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_resilience_llm_minor_change_variable_name(self):
        """
        Tests if the reconstructor can produce valid, structurally correct code
        when the LLM makes a minor variable name change in the reduced code.
        The content of the code line should reflect the LLM's change,
        while structure (indentation, comments) is preserved.
        """
        original_code = textwrap.dedent("""\
            def process_data(data_list):
                # Process each item in the list
                for item in data_list:
                    print(f"Processing {item}")
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)

        # Simulate LLM changing 'item' to 'element' in the reduced code line
        modified_reduced_code = reduced_code.replace("for item in data_list:", "for element in data_list:")
        modified_reduced_code = modified_reduced_code.replace('print(f"Processing {item}")', 'print(f"Processing {element}")')

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        # The reconstructed code should now contain 'element' but with original comments and whitespace.
        expected_reconstructed_code = textwrap.dedent("""\
            def process_data(data_list):
                # Process each item in the list
                for element in data_list:
                    print(f"Processing {element}")
            """)
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertEqual(len(reconstructor.discrepancies), 0) # No discrepancies expected for minor changes

    def test_resilience_llm_modifying_expression(self):
        """
        Tests if the reconstructor can produce valid, structurally correct code
        when the LLM modifies an expression in the reduced code.
        """
        original_code = textwrap.dedent("""\
            def calculate_area(length, width):
                # Formula for rectangle area
                area = length * width
                return area
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)

        # Simulate LLM changing the expression from '*' to '+'
        modified_reduced_code = reduced_code.replace("area = length * width", "area = length + width")

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        expected_reconstructed_code = textwrap.dedent("""\
            def calculate_area(length, width):
                # Formula for rectangle area
                area = length + width
                return area
            """)
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertEqual(len(reconstructor.discrepancies), 0) # No discrepancies expected for minor changes

    def test_empty_code(self):
        """Tests reconstruction of an empty string."""
        original_code = ""
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_code_with_only_comments_and_whitespace(self):
        """Tests reconstruction of code containing only comments and whitespace."""
        original_code = textwrap.dedent("""\
            # Just comments
            
            
            # Another comment
            
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_complex_nested_structure(self):
        """
        Tests byte-by-byte reconstruction of a complex nested structure
        including functions, loops, and conditionals.
        """
        original_code = textwrap.dedent("""\
            def outer_function(x):
                # Outer function doc
                def inner_function(y):
                    # Inner function doc
                    if y > 0:
                        for i in range(y):
                            print(f"Inner loop: {i}") # Loop print
                    else:
                        print("Y is not positive")
                    return y * 2
                
                result = inner_function(x + 5) # Call inner
                return result / 2
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_reconstruction_with_docstrings(self):
        """
        Tests that docstrings are correctly preserved during reconstruction,
        as they are part of the code structure.
        """
        original_code = textwrap.dedent("""\
            def example_function():
                \"\"\"This is a docstring.\"\"\"
                pass
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_reconstruction_with_trailing_whitespace_on_lines(self):
        """
        Tests that trailing whitespace on lines (if present in original)
        is correctly handled and restored.
        """
        original_code = textwrap.dedent("""\
            def func_with_trailing_space():   
                x = 10    
                # This comment has trailing space   
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)
        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(reduced_code)
        self.assertEqual(reconstructed_code, original_code)
        self.assertEqual(len(reconstructor.discrepancies), 0)

    def test_resilience_llm_adding_new_line_within_reduced_code(self):
        """
        Tests resilience when LLM adds a new line (e.g., a print statement)
        to the reduced code. The reconstructor should integrate it,
        and discrepancies should be signaled.
        """
        original_code = textwrap.dedent("""\
            def simple_func():
                a = 1
                b = 2
                return a + b
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)

        reduced_lines = reduced_code.splitlines()
        
        try:
            insert_idx = reduced_lines.index("b = 2") + 1
            modified_reduced_lines = reduced_lines[:insert_idx] + ["print('LLM added this!')"] + reduced_lines[insert_idx:]
            modified_reduced_code = "\n".join(modified_reduced_lines)
        except ValueError:
            self.fail("Could not find 'b = 2' in reduced code for modification.")

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        expected_reconstructed_code = textwrap.dedent("""\
            def simple_func():
                a = 1
                b = 2
                print('LLM added this!')
                return a + b
            """)
        
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertIn("Added: print('LLM added this!')", reconstructor.discrepancies)
        self.assertEqual(len(reconstructor.discrepancies), 1)

    def test_resilience_llm_removing_a_line(self):
        """
        Tests resilience when LLM removes an existing code line.
        The mock reconstructor should re-add the original line as a fallback
        and signal the discrepancy.
        """
        original_code = textwrap.dedent("""\
            def func_to_modify():
                line_one = 1
                line_two = 2
                line_three = 3
                return line_one + line_two + line_three
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)
        
        reduced_lines = reduced_code.splitlines()
        
        # Simulate LLM removing 'line_two = 2'
        modified_reduced_lines = [line for line in reduced_lines if line != "line_two = 2"]
        modified_reduced_code = "\n".join(modified_reduced_lines)

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        # The mock reconstructor re-adds the original line as a fallback
        expected_reconstructed_code = textwrap.dedent("""\
            def func_to_modify():
                line_one = 1
                line_two = 2
                line_three = 3
                return line_one + line_two + line_three
            """)
        
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertIn("Removed: line_two = 2", reconstructor.discrepancies)
        self.assertEqual(len(reconstructor.discrepancies), 1)

    def test_resilience_llm_adding_line_at_start(self):
        """
        Tests resilience when LLM adds a new line at the very beginning of the reduced code.
        """
        original_code = textwrap.dedent("""\
            def func():
                pass
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)

        modified_reduced_code = "import new_module\n" + reduced_code

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        expected_reconstructed_code = textwrap.dedent("""\
            import new_module
            def func():
                pass
            """)
        
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertIn("Added: import new_module", reconstructor.discrepancies)
        self.assertEqual(len(reconstructor.discrepancies), 1)

    def test_resilience_llm_adding_line_at_end(self):
        """
        Tests resilience when LLM adds a new line at the very end of the reduced code.
        """
        original_code = textwrap.dedent("""\
            def func():
                pass
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)

        modified_reduced_code = reduced_code + "\nprint('Done!')"

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        expected_reconstructed_code = textwrap.dedent("""\
            def func():
                pass
            print('Done!')
            """)
        
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertIn("Added (end): print('Done!')", reconstructor.discrepancies)
        self.assertEqual(len(reconstructor.discrepancies), 1)

    def test_resilience_llm_modifying_and_adding_lines(self):
        """
        Tests resilience when LLM both modifies an existing line and adds a new line.
        """
        original_code = textwrap.dedent("""\
            def complex_func(x, y):
                # Initial calculation
                result = x * y
                # End of function
                return result
            """)
        reduced_code, metadata = self.reducer.reduce(original_code)

        reduced_lines = reduced_code.splitlines()
        
        # Simulate LLM changing 'result = x * y' to 'result = x + y'
        # And adding 'print("Modified calculation!")' after it
        modified_reduced_lines = []
        for line in reduced_lines:
            if line == "result = x * y":
                modified_reduced_lines.append("result = x + y") # Modified
                modified_reduced_lines.append("print('Modified calculation!')") # Added
            else:
                modified_reduced_lines.append(line)
        
        modified_reduced_code = "\n".join(modified_reduced_lines)

        reconstructor = MockReconstructor(metadata)
        reconstructed_code = reconstructor.reconstruct(modified_reduced_code)

        expected_reconstructed_code = textwrap.dedent("""\
            def complex_func(x, y):
                # Initial calculation
                result = x + y
                print('Modified calculation!')
                # End of function
                return result
            """)
        
        self.assertEqual(reconstructed_code, expected_reconstructed_code)
        self.assertIn("Added: print('Modified calculation!')", reconstructor.discrepancies)
        self.assertEqual(len(reconstructor.discrepancies), 1) # Only one added line, modification is not a discrepancy


if __name__ == '__main__':
    unittest.main()