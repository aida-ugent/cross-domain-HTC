import os
import re
from typing import List, Tuple, Dict

def parse_result_file(file_path: str) -> Tuple[List[float], List[float]]:
    with open(file_path, 'r') as f:
        content = f.read()

    prec_match = re.search(r'prec\s*=\s*([\d\.\s]+)', content)
    recall_match = re.search(r'recall\s*=\s*([\d\.\s]+)', content)

    if not prec_match or not recall_match:
        raise ValueError(f"Could not parse data from {file_path}")

    prec = [float(x) for x in prec_match.group(1).split()]
    recall = [float(x) for x in recall_match.group(1).split()]

    return prec, recall

def calculate_average_results(base_dir: str, name: str) -> Dict[str, Dict[str, Tuple[List[float], List[float], List[int]]]]:
    results = {}

    for dir_name in os.listdir(base_dir):
        match = re.match(r'(.+)_seed(\d+)', dir_name)
        if match:
            model_name, seed = match.groups()
            if model_name != name:
                continue
            file_path = os.path.join(base_dir, dir_name, 'result.log')
            if os.path.exists(file_path):
                prec, recall = parse_result_file(file_path)
                
                if model_name not in results:
                    results[model_name] = {
                        'prec': [prec],
                        'recall': [recall],
                        'seeds': [int(seed)]
                    }
                else:
                    results[model_name]['prec'].append(prec)
                    results[model_name]['recall'].append(recall)
                    results[model_name]['seeds'].append(int(seed))

    return results

def calculate_averages(results: Dict[str, Dict[str, List[List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    averages = {}
    for model, data in results.items():
        avg_prec = [sum(x) / len(x) for x in zip(*data['prec'])]
        avg_recall = [sum(x) / len(x) for x in zip(*data['recall'])]
        averages[model] = {
            'prec': avg_prec,
            'recall': avg_recall,
            'seeds': data['seeds']
        }
    return averages

def main(base_dir: str, name: str = 'bert'):
    try:
        results = calculate_average_results(base_dir, name)
        averages = calculate_averages(results)

        for model, data in averages.items():
            print(f"==== average evaluation results for {model} ====")
            print(f"prec   = {' '.join(f'{x:.2f}' for x in data['prec'])}")
            print(f"recall = {' '.join(f'{x:.2f}' for x in data['recall'])}")
            print(f"seeds  = {', '.join(map(str, sorted(data['seeds'])))}")
            print()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='The base directory containing the result logs')
    parser.add_argument('--name', help='The name of the model to calculate averages for', default='bert')
    args = parser.parse_args()
    main(args.base_dir, args.name)