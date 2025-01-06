import fire

from utils.misc import load_jsonl, save_json


def postprocess_gsm8k_official(data_path, save_path):
    data = load_jsonl(data_path)
    result = []
    for iid, d in enumerate(data):
        postprocessed = {
            "id": iid,
            "problem": d["question"],
            "solution": d["answer"],
        }

        postprocessed['solution'] = postprocessed['solution'].replace('####', 'The answer is:')
        result.append(postprocessed)

    save_json(result, save_path)


if __name__ == "__main__":
    fire.Fire(postprocess_gsm8k_official)
