import re
import os
import subprocess

def convert_text(text):
    """Converts text to lowercase and tokenizes."""
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

def eval_meteor_test_webnlg(folder_data, pred_file, dataset):
    """Evaluate METEOR score using the meteor-1.5.jar script."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = os.path.join(dir_path, "../utils")
    cmd_string = [
        "java", "-jar", os.path.join(folder_data_before, "meteor-1.5.jar"),
        pred_file, os.path.join(folder_data, f"{dataset}.target_eval_meteor"),
        "-l", "en", "-norm", "-r", "3"
    ]

    try:
        subprocess.run(cmd_string, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error while evaluating METEOR: {e.stderr}")
    
    with open(pred_file.replace("txt", "meteor"), 'r') as file:
        meteor_info = file.readlines()[-1].strip()

    return meteor_info

def eval_chrf_test_webnlg(folder_data, pred_file, dataset):
    """Evaluate CHRF score using the chrf++.py script."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = os.path.join(dir_path, "../utils")
    cmd_string = [
        "python", os.path.join(folder_data_before, "chrf++.py"),
        "-H", pred_file,
        "-R", os.path.join(folder_data, f"{dataset}.target_eval_crf")
    ]

    try:
        result = subprocess.run(cmd_string, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error while evaluating CHRF: {e.stderr}")

    output_lines = result.stdout.splitlines()
    chrf_info_1 = output_lines[1].strip()
    chrf_info_2 = output_lines[2].strip()

    return f"{chrf_info_1} {chrf_info_2}"

import os
import tempfile
import subprocess

def create_temp_file(content, dir="/tmp"):
    # Remove empty lines and strip whitespace
    if isinstance(content, list):
        content = [line.strip() for line in content if line.strip()]
    else:
        content = "\n".join([line.strip() for line in content.splitlines() if line.strip()])

    temp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', dir=dir)
    temp.write(content)
    temp.close()
    return temp.name

# Utility to clean up temporary files
def cleanup_temp_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

import re

def eval_bleu(ref_file_content, pred_file_content, temp_dir="/tmp"):
    # Ensure content is a string; if it's a list, join with newline
    if isinstance(ref_file_content, list):
        ref_file_content = "\n".join(ref_file_content)
    if isinstance(pred_file_content, list):
        pred_file_content = "\n".join(pred_file_content)

    # Create temporary files for reference and prediction content
    ref_file = create_temp_file(ref_file_content, dir=temp_dir)
    pred_file = create_temp_file(pred_file_content, dir=temp_dir)

    # Define the BLEU script path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = os.path.join(dir_path, "utils")
    bleu_script = os.path.join(folder_data_before, "multi-bleu.perl")

    cmd_string = [
        "perl", bleu_script, "-lc", ref_file
    ]

    try:
        # Run the BLEU script with predictions as input
        result = subprocess.run(
            cmd_string, check=True, input=open(pred_file).read(),
            text=True, capture_output=True
        )
        bleu_output = result.stdout
        # print(f"BLEU raw output:\n{bleu_output}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error while evaluating BLEU: {e.stderr}")
    finally:
        # Cleanup temporary files
        cleanup_temp_files(ref_file, pred_file)

    # Extract the BLEU score using regex
    match = re.search(r"BLEU = ([\d.]+)", bleu_output)
    if match:
        return float(match.group(1))  # Return the BLEU score as a float

    # If BLEU score not found
    print("Unable to find BLEU score in output.")
    return 0.0  # Return 0.0 as a default value


def eval_meteor(ref_file_content, pred_file_content, temp_dir="/tmp"):
    # Ensure inputs are lists
    if not isinstance(ref_file_content, list):
        ref_file_content = ref_file_content.split('\n')
    if not isinstance(pred_file_content, list):
        pred_file_content = pred_file_content.split('\n')
    
    # Clean and filter the content
    ref_lines = [line.strip() for line in ref_file_content if line.strip()]
    pred_lines = [line.strip() for line in pred_file_content if line.strip()]
    
    # Print lengths for debugging
    print(f"Reference lines: {len(ref_lines)}")
    print(f"Prediction lines: {len(pred_lines)}")
    
    # Handle length mismatch
    if len(ref_lines) != len(pred_lines):
        print("Warning: Mismatch in number of lines. Truncating to shorter length.")
        min_len = min(len(ref_lines), len(pred_lines))
        ref_lines = ref_lines[:min_len]
        pred_lines = pred_lines[:min_len]
    
    # Create temporary files
    ref_file = create_temp_file("\n".join(ref_lines), dir=temp_dir)
    pred_file = create_temp_file("\n".join(pred_lines), dir=temp_dir)
    
    try:
        # Rest of your existing code...
        dir_path = os.path.dirname(os.path.realpath(__file__))
        meteor_jar_path = os.path.join(dir_path, "utils/meteor-1.5.jar")
        cmd_string = ["java", "-jar", meteor_jar_path, pred_file, ref_file]
        
        result = subprocess.run(cmd_string, check=True, capture_output=True, text=True)
        meteor_output = result.stdout
        
        for line in meteor_output.split("\n"):
            if "Final score:" in line:
                return float(line.split()[-1])
        
        return 0.0  # Return 0 if no score found
        
    except Exception as e:
        print(f"Error in METEOR evaluation: {str(e)}")
        return 0.0
    finally:
        cleanup_temp_files(ref_file, pred_file)


def eval_chrf(ref_file_content, pred_file_content, temp_dir="/tmp"):
    # Ensure content is a string; if it's a list, join with newline
    if isinstance(ref_file_content, list):
        ref_file_content = "\n".join(ref_file_content)
    if isinstance(pred_file_content, list):
        pred_file_content = "\n".join(pred_file_content)

    # Create temporary files for reference and prediction content
    ref_file = create_temp_file(ref_file_content, dir=temp_dir)
    pred_file = create_temp_file(pred_file_content, dir=temp_dir)

    # Define the CHRF++ script path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = os.path.join(dir_path, "utils")
    chrf_script = os.path.join(folder_data_before, "chrf++.py")

    cmd_string = [
        "python", chrf_script,
        "-H", pred_file, "-R", ref_file
    ]

    try:
        # Run the CHRF++ script
        result = subprocess.run(cmd_string, check=True, capture_output=True, text=True)
        chrf_output = result.stdout
        # print(f"CHRF raw output:\n{chrf_output}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error while evaluating CHRF: {e.stderr}")
    finally:
        # Cleanup temporary files
        cleanup_temp_files(ref_file, pred_file)

    # Extract the main CHRF score (e.g., '23.8992') from the output
    try:
        for line in chrf_output.splitlines():
            if line.startswith("c6+w2-F2"):
                return float(line.split("\t")[1])  # Extract the numerical CHRF score
    except (IndexError, ValueError):
        pass

    # If CHRF score not found, return 0.0
    print("Unable to find CHRF score in output.")
    return 0.0