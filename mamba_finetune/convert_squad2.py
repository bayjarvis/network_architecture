# https://rajpurkar.github.io/SQuAD-explorer/

import json

def convert_json_to_jsonl(input_file, output_file):
    # Load the original JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Iterate through the data
        for entry in data['data']:
            for paragraph in entry['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    prompt = qa['question']
                    # Use the first answer as the response, or an empty string if no answer
                    response = qa['answers'][0]['text'] if qa['answers'] else ''
                    # Create a dictionary for the JSONL entry
                    jsonl_entry = {"context": context, "prompt": prompt, "response": response}
                    # Write the entry as a JSON line
                    json.dump(jsonl_entry, outfile)
                    outfile.write('\n')

if __name__ == "__main__":
    # Example usage
    convert_json_to_jsonl('data/train_squad2.0.json', 'data/train_squad2.0.jsonl')
    convert_json_to_jsonl('data/dev_squad2.0.json', 'data/dev_squad2.0.jsonl')
