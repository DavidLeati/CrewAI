import unittest
import textwrap
from src.llm_token_optimizer.api import reduce_code, reconstruct_code

class TestEndToEnd(unittest.TestCase):

    def test_empty_code_e2e(self):
        """
        Test the end-to-end process with an empty code string.
        """
        original_code = ""
        reduced_code = reduce_code(original_code)
        reconstructed_code = reconstruct_code(reduced_code)
        self.assertEqual(original_code, reconstructed_code)

    def test_code_with_only_comments_e2e(self):
        """
        Test the end-to-end process with code containing only comments.
        Expect the original code to be reconstructed byte-by-byte if the
        reconstructor aims for full structural preservation, or an empty
        string if comments are always discarded. Given the 'total resilience'
        goal, we expect comments to be preserved.
        """
        original_code = textwrap.dedent("""\
            # This is a comment
                # Another comment
            """)
        reduced_code = reduce_code(original_code)
        reconstructed_code = reconstruct_code(reduced_code)
        # Assuming the reconstructor will restore comments and original whitespace
        self.assertEqual(original_code, reconstructed_code)

    def test_simple_function_e2e(self):
        """
        Test a simple function definition through the full reduction-reconstruction cycle.
        Expect byte-by-byte reconstruction, including comments and original formatting.
        """
        original_code = textwrap.dedent("""\
            def hello_world():
                # A simple print statement
                print("Hello, World!")
            """)
        
        reduced_code = reduce_code(original_code)
        reconstructed_code = reconstruct_code(reduced_code)
        self.assertEqual(original_code, reconstructed_code)

    def test_code_with_comments_and_whitespace_e2e(self):
        """
        Test code containing comments and varying whitespace.
        Expect the original code to be reconstructed accurately,
        preserving comments and original whitespace.
        """
        original_code = textwrap.dedent("""\
            # This is a comment
            def calculate_sum(a, b):  # Another comment
                result = a + b   # Add two numbers
                return result

            # End of file
            """)
        
        reduced_code = reduce_code(original_code)
        reconstructed_code = reconstruct_code(reduced_code)
        self.assertEqual(original_code, reconstructed_code)

    def test_complex_code_e2e(self):
        """
        Test a more complex code snippet, including imports, classes, methods,
        and multiple lines of logic, through the full cycle.
        Ensure byte-by-byte reconstruction, including comments and original formatting.
        """
        original_code = textwrap.dedent("""\
            import os
            import sys

            class MyClass:
                def __init__(self, name):
                    self.name = name

                def greet(self):
                    # This is a greeting method
                    if self.name:
                        print(f"Hello, {self.name}!")
                    else:
                        print("Hello, anonymous!")

            def main():
                obj = MyClass("World")
                obj.greet()
                # Another line
                print("Program finished.")

            if __name__ == "__main__":
                main()
            """)
        
        reduced_code = reduce_code(original_code)
        reconstructed_code = reconstruct_code(reduced_code)
        self.assertEqual(original_code, reconstructed_code)

    # --- New E2E Resilience Tests for LLM Modifications ---

    def test_e2e_resilience_llm_modifying_line(self):
        """
        Tests end-to-end resilience when LLM modifies an existing code line.
        The reconstructed code should reflect the LLM's change while preserving structure.
        """
        original_code = textwrap.dedent("""\
            def calculate(a, b):
                # Perform multiplication
                result = a * b
                return result
            """)
        
        reduced_code = reduce_code(original_code)
        
        # Simulate LLM changing 'result = a * b' to 'result = a + b'
        modified_reduced_code = reduced_code.replace("result = a * b", "result = a + b")
        
        reconstructed_code = reconstruct_code(modified_reduced_code)
        
        expected_code = textwrap.dedent("""\
            def calculate(a, b):
                # Perform multiplication
                result = a + b
                return result
            """)
        
        self.assertEqual(expected_code, reconstructed_code)

    def test_e2e_resilience_llm_adding_line_in_middle(self):
        """
        Tests end-to-end resilience when LLM adds a new line in the middle of code.
        The reconstructed code should integrate the new line with appropriate indentation.
        """
        original_code = textwrap.dedent("""\
            def simple_process():
                step_one()
                step_two()
            """)
        
        reduced_code = reduce_code(original_code)
        
        # Simulate LLM adding 'print("Intermediate step")' after 'step_one()'
        reduced_lines = reduced_code.splitlines()
        try:
            insert_idx = reduced_lines.index("step_one()") + 1
            modified_reduced_lines = reduced_lines[:insert_idx] + ["print('Intermediate step')"] + reduced_lines[insert_idx:]
            modified_reduced_code = "\n".join(modified_reduced_lines)
        except ValueError:
            self.fail("Could not find 'step_one()' in reduced code for modification.")

        reconstructed_code = reconstruct_code(modified_reduced_code)
        
        expected_code = textwrap.dedent("""\
            def simple_process():
                step_one()
                print('Intermediate step')
                step_two()
            """)
        
        self.assertEqual(expected_code, reconstructed_code)

    def test_e2e_resilience_llm_removing_line(self):
        """
        Tests end-to-end resilience when LLM removes an existing code line.
        The reconstructed code should omit the removed line while preserving the rest of the structure.
        """
        original_code = textwrap.dedent("""\
            def cleanup_resources():
                close_file_handles()
                release_memory() # This line will be removed
                log_completion()
            """)
        
        reduced_code = reduce_code(original_code)
        
        # Simulate LLM removing 'release_memory()'
        modified_reduced_code = "\n".join([line for line in reduced_code.splitlines() if line != "release_memory()"])
        
        reconstructed_code = reconstruct_code(modified_reduced_code)
        
        # Expected: the line is simply gone, and structure is preserved for others.
        expected_code = textwrap.dedent("""\
            def cleanup_resources():
                close_file_handles()
                log_completion()
            """)
        
        self.assertEqual(expected_code, reconstructed_code)

    def test_e2e_resilience_llm_adding_line_at_start_and_end(self):
        """
        Tests end-to-end resilience when LLM adds lines at the beginning and end of the code.
        """
        original_code = textwrap.dedent("""\
            def core_logic():
                do_something()
            """)
        
        reduced_code = reduce_code(original_code)
        
        modified_reduced_code = "import new_lib\n" + reduced_code + "\nfinal_log()"
        
        reconstructed_code = reconstruct_code(modified_reduced_code)
        
        expected_code = textwrap.dedent("""\
            import new_lib
            def core_logic():
                do_something()
            final_log()
            """)
        
        self.assertEqual(expected_code, reconstructed_code)

    def test_e2e_resilience_llm_multiple_changes(self):
        """
        Tests end-to-end resilience with a combination of LLM modifications:
        modifying a line, adding a line, and removing a line.
        """
        original_code = textwrap.dedent("""\
            def process_workflow(data):
                # Step 1: Initialize
                initialized_data = data + 10
                # Step 2: Process
                processed_data = initialized_data * 2 # This line will be modified
                # Step 3: Log intermediate
                log_data(processed_data) # This line will be removed
                # Step 4: Finalize
                final_result = processed_data - 5
                return final_result
            """)
        
        reduced_code = reduce_code(original_code)
        
        reduced_lines = reduced_code.splitlines()
        modified_reduced_lines = []
        
        for line in reduced_lines:
            if line == "processed_data = initialized_data * 2":
                modified_reduced_lines.append("processed_data = initialized_data / 2") # Modified
                modified_reduced_lines.append("print('Intermediate check done.')") # Added after modification
            elif line == "log_data(processed_data)":
                # This line is removed
                pass
            else:
                modified_reduced_lines.append(line)
        
        modified_reduced_code = "\n".join(modified_reduced_lines)
        
        reconstructed_code = reconstruct_code(modified_reduced_code)
        
        expected_code = textwrap.dedent("""\
            def process_workflow(data):
                # Step 1: Initialize
                initialized_data = data + 10
                # Step 2: Process
                processed_data = initialized_data / 2
                print('Intermediate check done.')
                # Step 4: Finalize
                final_result = processed_data - 5
                return final_result
            """)
        
        self.assertEqual(expected_code, reconstructed_code)


if __name__ == '__main__':
    unittest.main()