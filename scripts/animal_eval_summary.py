import json
from sl.utils import module_utils, file_utils


#eval_file_name = "qwen_owl_eval.jsonl"
eval_file_name = "llama3.1-8b-it_owl_eval.json"
animal = "owl"

if __name__ == "__main__":
    prefs = []
    with open(f"./data/preference_numbers/owl/{eval_file_name}") as f:
        for line in f:
            prefs.append(json.loads(line))
    
    print(len(prefs))
    print(prefs[0].keys())


    count, ani_count = 0, 0
    for i, q in enumerate(prefs):
        q_ani_count = 0
        for j, resp in enumerate(q['responses']):
            comp = resp['response']['completion']
            count += 1
            #print(f"\t\t{repr(comp)}")
            if animal in comp.lower():
                ani_count += 1
                q_ani_count += 1
        print(f"[{i}]    {repr(q['question'])} ({q_ani_count}/{len(q['responses'])})")

    print(f"{ani_count}/{count} ({ani_count/count:.3f})")
    