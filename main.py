#!/usr/bin/env python
import argparse
import importlib
import os
import timeit

def find_closest_file(string):
    # Get the list of files in the experiments directory
    files = os.listdir("./experiments")
    string = string.replace(".py", "")
    # Initialize variables to store the closest file name and its similarity score
    closest_file = None
    max_similarity = 0

    # Iterate over the files
    for file in files:
        # Calculate the similarity score between the string and the file name
        file = file.replace(".py", "")
        similarity = len(set(string) & set(file))

        # Update the closest file and similarity score if the current file has a higher similarity score
        if similarity > max_similarity:
            closest_file = file
            max_similarity = similarity

    # Check if the closest file is different from the chosen file
    if closest_file != string:
        print(
            f"Warning: The closest file found is '{closest_file}', which is different from the chosen file '{string}'."
        )

    return closest_file


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Description of your program.")
    # Add arguments using add_argument() method
    parser.add_argument("-f", "--file", default="", type=str, help="Path to the file.")
    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the values of the arguments
    file_path = args.file

    if not file_path:
        raise ValueError("File path cannot be empty.")

    file_path = find_closest_file(file_path)

    # Dynamically import the main function from the given file
    # module_name = file_path.replace('/', '.').replace('.py', '')
    module_name = file_path
    module = importlib.import_module(f"experiments.{module_name}.{module_name}")
    main_function = getattr(module, "main")

    # Call the main function
    main_function()


if __name__ == "__main__":
    main()
