from flask import Flask, request, jsonify
import subprocess
from pathlib import Path
import json
import re
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

DATA_DIR = Path("./data")

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\shant\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


@app.route('/run', methods=['POST'])
def run_task():
    task_description = request.args.get('task')
    if not task_description:
        return jsonify({"error": "Task description is required."}), 400

    try:
        print(f"Task Description: {task_description}")  # Debugging output

        # Handle Task A1: datagen
        if "datagen" in task_description:
            email = task_description.split("email=")[-1].strip()
            execute_datagen(email)
            return jsonify({"status": "success", "message": "Data generation completed."}), 200

        # Handle Task A2: Format Markdown
        if "format" in task_description.lower() and "markdown" in task_description.lower():
            execute_task("format_markdown")
            return jsonify({"status": "success", "message": "Markdown file formatted successfully."}), 200

        # Handle Task A3: Count Wednesdays
        if "count wednesdays" in task_description.lower():
            message = execute_task("count_wednesdays")
            return jsonify({"status": "success", "message": message}), 200

        # Handle Task A4: Sort Contacts
        if "sort contacts" in task_description.lower():
            message = execute_task("sort_contacts")
            return jsonify({"status": "success", "message": message}), 200

        # Handle Task A5: Extract Recent Logs
        if "extract" in task_description.lower() and "log" in task_description.lower():
            message = execute_task("extract_recent_logs")
            return jsonify({"status": "success", "message": message}), 200

        # Handle Task A6: Create Markdown Index
        if "create markdown index" in task_description.lower():
            message = execute_task("create_markdown_index")
            return jsonify({"status": "success", "message": message}), 200
        
        # Handle Task A7: Extract Email Address
        if "extract sender email" in task_description.lower():
            input_file = DATA_DIR / "email.txt"
            output_file = DATA_DIR / "email-sender.txt"
            extract_sender_email(input_file, output_file)
            print(f"Received Task: {task_description}")
            return jsonify({"status": "success", "message": "Sender's email address extracted and saved."}), 200
        
        # Handle Task A8: Extract Credit Card Number
        if "extract credit card" in task_description.lower():
            try:
                image_path = DATA_DIR / "credit_card.png"
                if not image_path.exists():
                    raise FileNotFoundError(f"{image_path} not found.")
                
                text = pytesseract.image_to_string(image_path)
    
                # Regex to extract credit card number
                import re
                match = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', text)
                if match:
                    card_number = match.group(0).replace(" ", "")
                    output_path = DATA_DIR / "credit-card.txt"
                    with open(output_path, "w") as f:
                        f.write(card_number)
                    return jsonify({"status": "success", "message": "Credit card number extracted and saved."}), 200
                else:
                    raise ValueError("No valid credit card number found in the image.")
            except Exception as e:
                return jsonify({"error": f"Internal Server Error: Failed to extract credit card number: {str(e)}"}), 500

        # Handle Task A9: Find Most Similar Comments
        if "find similar comments" in task_description.lower():
            input_path = DATA_DIR / "comments.txt"
            output_path = DATA_DIR / "comments-similar.txt"
            message = find_most_similar_comments(input_path, output_path)
            return jsonify({"status": "success", "message": message}), 200
        
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


def find_most_similar_comments(input_path, output_path):
    """Find the most similar pair of comments using embeddings."""
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    try:
        # Load comments from file
        with open(input_path, "r", encoding="utf-8") as f:
            comments = [line.strip() for line in f.readlines() if line.strip()]

        if len(comments) < 2:
            raise ValueError("Not enough comments to find similarity.")

        # Load the embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Compute embeddings
        embeddings = model.encode(comments, convert_to_tensor=True)

        # Find the most similar pair
        max_sim = -1
        best_pair = None

        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                sim_score = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                if sim_score > max_sim:
                    max_sim = sim_score
                    best_pair = (comments[i], comments[j])

        if best_pair is None:
            raise ValueError("No similar comments found.")

        # Write the most similar comments to output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(best_pair))

        return f"Most similar comments saved to {output_path}."
    except Exception as e:
        raise RuntimeError(f"Failed to find similar comments: {e}")


def extract_sender_email(input_path, output_path):
    """Extract the sender's email address from an email file."""
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    try:
        # Read the email content
        with open(input_path, 'r') as file:
            email_content = file.read()

        # Extract the sender's email using regex
        match = re.search(r'From:.*<(.+?)>', email_content)
        if match:
            sender_email = match.group(1)
            # Write the sender's email to the output file
            with open(output_path, 'w') as output_file:
                output_file.write(sender_email)
            print(f"Sender's email address extracted: {sender_email}")
        else:
            raise ValueError("Sender's email address not found in the email content.")
    except Exception as e:
        print(f"Error extracting sender email: {e}")
        raise RuntimeError(f"Failed to extract sender email: {e}")

def execute_datagen(email):
    """Run the datagen.py script with the provided email."""
    script_path = DATA_DIR / "datagen.py"
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path} not found. Please ensure the file exists.")

    try:
        python_path = Path("venv/Scripts/python.exe").resolve()  # Full path to Python
        result = subprocess.run(
            [str(python_path), str(script_path), email, "--root", str(DATA_DIR)],
            check=True,
            capture_output=True,  # Capture stdout and stderr for debugging
            text=True,
        )
        print("Datagen Output:", result.stdout)  # Log script output
    except subprocess.CalledProcessError as e:
        print("Datagen Error:", e.stderr)  # Log script error output
        raise e


def execute_task(action):
    """Execute specific tasks based on the action."""
    if action == "format_markdown":
        # File to format
        md_file = DATA_DIR / "format.md"
        if not md_file.exists():
            raise FileNotFoundError(f"{md_file} not found.")
        
        # Full path to Prettier
        prettier_path = r"C:\Users\shant\AppData\Roaming\npm\prettier.cmd"  # Full path to Prettier

        # Format file using Prettier
        try:
            subprocess.run([prettier_path, "--write", str(md_file)], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Prettier failed: {e}")
        except FileNotFoundError as e:
            raise RuntimeError(f"Prettier executable not found: {e}")
    # else:
    #     raise ValueError(f"Unsupported action: {action}")
    
    """Execute specific tasks based on the action."""
    if action == "count_wednesdays":
        # File paths
        dates_file = DATA_DIR / "dates.txt"
        output_file = DATA_DIR / "dates-wednesdays.txt"

        # Ensure the input file exists
        if not dates_file.exists():
            raise FileNotFoundError(f"{dates_file} not found.")

        try:
            # Read the dates and count Wednesdays
            with open(dates_file, "r") as f:
                dates = f.readlines()

            from datetime import datetime

            count = 0
            for date_str in dates:
                try:
                    # Parse the date and check if it's a Wednesday
                    date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
                    if date.weekday() == 2:  # 2 = Wednesday
                        count += 1
                except ValueError:
                    # Skip invalid date formats
                    continue

            # Write the count to the output file
            with open(output_file, "w") as f:
                f.write(str(count))

            return f"Counted {count} Wednesdays in {dates_file}."
        except Exception as e:
            raise RuntimeError(f"Failed to count Wednesdays: {e}")
        
    """Execute specific tasks based on the action."""
    if action == "sort_contacts":
        # File paths
        input_file = DATA_DIR / "contacts.json"
        output_file = DATA_DIR / "contacts-sorted.json"

        # Ensure the input file exists
        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} not found.")

        try:
            # Read the contacts
            with open(input_file, "r") as f:
                contacts = json.load(f)

            # Sort the contacts by last_name and then by first_name
            sorted_contacts = sorted(
                contacts, key=lambda c: (c["last_name"].lower(), c["first_name"].lower())
            )

            # Write the sorted contacts to the output file
            with open(output_file, "w") as f:
                json.dump(sorted_contacts, f, indent=4)

            return f"Sorted {len(contacts)} contacts in {input_file} and saved to {output_file}."
        except Exception as e:
            raise RuntimeError(f"Failed to sort contacts: {e}")
        
    """Execute specific tasks based on the action."""
    if action == "extract_recent_logs":
        # Directory and output file paths
        logs_dir = DATA_DIR / "logs"
        output_file = DATA_DIR / "logs-recent.txt"

        # Ensure the logs directory exists
        if not logs_dir.exists() or not logs_dir.is_dir():
            raise FileNotFoundError(f"{logs_dir} directory not found.")

        try:
            # Get all .log files sorted by modification time (most recent first)
            log_files = sorted(
                logs_dir.glob("*.log"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Extract the first line from the 10 most recent files
            extracted_lines = []
            for log_file in log_files[:10]:  # Only process the top 10 files
                with open(log_file, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:  # Only add non-empty lines
                        extracted_lines.append(first_line)

            # Write the extracted lines to the output file
            with open(output_file, "w") as f:
                f.write("\n".join(extracted_lines))

            return f"Extracted {len(extracted_lines)} lines from the 10 most recent log files in {logs_dir}."
        except Exception as e:
            raise RuntimeError(f"Failed to extract recent logs: {e}")
    
    """Execute specific tasks based on the action."""
    if action == "create_markdown_index":
        docs_dir = DATA_DIR / "docs"
        output_file = docs_dir / "index.json"

        print(f"DATA_DIR: {DATA_DIR}")
        print(f"Looking for Markdown files in: {docs_dir}")

        if not docs_dir.exists() or not docs_dir.is_dir():
            raise FileNotFoundError(f"{docs_dir} directory not found.")

        try:
            index = {}

            # Debug: Track processed files
            print(f"Markdown files found:")
            for md_file in docs_dir.rglob("*.md"):
                print(f"- Processing: {md_file}")

                with open(md_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    print(f"Content of {md_file}:")
                    print("".join(lines))  # Log file content

                    for line in lines:
                        if line.strip().startswith("#"):
                            title = line.strip().lstrip("#").strip()
                            index[md_file.name] = title
                            break
                    else:
                        # Handle files without headers
                        print(f"No header found in {md_file}. Assigning filename as title.")
                        index[md_file.name] = md_file.stem  # Use filename as title

            print(f"Generated index: {index}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=4)

            return f"Created Markdown index for {len(index)} files in {docs_dir}."
        except Exception as e:
            raise RuntimeError(f"Failed to create Markdown index: {e}")
    
    if action == "extract_credit_card":
        # File paths
        image_path = DATA_DIR / "credit_card.png"  # Corrected file name
        output_file = DATA_DIR / "credit_card.txt"

        # Ensure the input file exists
        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} not found.")

        try:
            from PIL import Image
            import pytesseract  # Ensure pytesseract is installed

            # Open the image and apply OCR
            image = Image.open(input_file)
            text = pytesseract.image_to_string(image_path)
            print("Extracted Text from Tesseract:", text)


            # Extract the credit card number (assuming a 16-digit number)
            match = re.search(r"\b\d{4} \d{4} \d{4} \d{4}\b", text)
            if not match:
                raise ValueError("No valid credit card number found in the image.")

            # Format the number (remove spaces)
            credit_card_number = match.group(0).replace(" ", "")

            # Write the result to the output file
            with open(output_file, "w") as f:
                f.write(credit_card_number)

            return f"Extracted credit card number and saved to {output_file}."
        except Exception as e:
            raise RuntimeError(f"Failed to extract credit card number: {e}")
    
    else:
        raise ValueError(f"Unsupported action: {action}")


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the LLM Automation Agent API! Use /run or /read endpoints."

@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({"error": "File path is required."}), 400

    try:
        full_path = DATA_DIR / file_path
        if not full_path.exists():
            return jsonify({"error": "File not found."}), 404

        with open(full_path, 'r') as file:
            content = file.read()

        return content, 200
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
