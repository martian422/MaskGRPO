import numpy as np
import re
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer
from transformers import CLIPModel, CLIPProcessor, AutoModel
from torchvision import transforms

import os
import huggingface_hub
import torch
from torchvision.transforms.functional import to_pil_image

from PIL import Image
from collections import defaultdict

from code_utils.code_providers import get_provider

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_responses = []
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
        
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    extracted_responses = []
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func_mmada(completions, **kwargs) -> list[float]:
    # abandoned version. currently we use llada style.
    pattern = r"<think>.*?</think>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]  
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>") == 1:
        count += 0.125
    # if text.count("\n<answer>\n") == 1:
    #     count += 0.125
    #     count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    # if text.count("\n</answer>") == 1:
    #     count += 0.125
    #     count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


def reward_clip_score(device="cuda"):
    
    # CLIP
    print("Loading CLIP model...")
    print("You may need to set HF_ENDPOINT for first time usage.")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def clip_score(completions, prompts, **kwargs):

        with torch.no_grad():
            completions = torch.clamp((completions + 1.0) / 2.0, min=0.0, max=1.0)
            
            inputs = clip_processor(text=prompts, images=completions,
                                    padding=True, 
                                    truncation=True,
                                    max_length=77,
                                    return_tensors="pt", 
                                    do_rescale=False, do_resize=True, do_center_crop=False).to(device)
            img_fts = clip_model.get_image_features(inputs['pixel_values'])
            gen_embeds = img_fts
            
            # CLIP DIR
            text_embeddings = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            text_gen = text_embeddings[:completions.shape[0]]
            text_gen /= text_gen.norm(-1, keepdim=True)
            gen_embeds /= gen_embeds.norm(-1, keepdim=True)
            clip_dir_persample = torch.nn.functional.cosine_similarity(text_gen, gen_embeds)

        return clip_dir_persample
    
    return clip_score

def hpsv2(device, hps_version="v2.1"):
    """
    Prepares the function for computing Human Preference Score (HPSv2) between images and text prompts.

    Args
    ----
    device : torch.device
        The device (CPU or GPU) on which to perform the computation.
    hps_version: str, optional
        Version of the HPS model to use. Options are "v2.0" or "v2.1". 
        Defaults to "v2.1".
    
    Returns
    -------
    callable
        A function that takes:
            - images (list): A list of torch.tensor images to evaluate
            - prompts (list): A list of text prompts corresponding to the images
        And returns:
            - torch.Tensor: HPS scores between the images and associated prompts
    """
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path, hps_version_map

    # Initialize the model
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    def hpsv2_score(image, prompt):
        with torch.no_grad():
            # Process the image
            image = to_pil_image(torch.clamp((image + 1) / 2, 0, 1))
            image = preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.amp.autocast('cuda'):
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return [hps_score[0]]

    def _hpsv2_score(prompts, completions, **kwargs):
        with torch.no_grad():
            result = [hpsv2_score(completions[i], prompts[i]) for i in range(len(completions))] # whether you shall multiply it 5 times?

        return torch.tensor(result).squeeze().cpu()

    return _hpsv2_score

def hpsv3(device='cuda'):
    """
    Returns a function to compute HPSv3 reward scores given image tensors and prompts.
    Running locally will consume a lot memory.
    We suggest to run it as a remote service.

    Args
    ----
    device : str or torch.device
        The device to run the model on.

    Returns
    -------
    callable
        Function that takes:
            - images (list of torch.Tensor): Images in [-1,1], shape (C,H,W)
            - prompts (list of str): Corresponding text prompts
        Returns:
            - torch.Tensor: HPSv3 scores
    """
    from hpsv3 import HPSv3RewardInferencer
    import tempfile

    inferencer = HPSv3RewardInferencer(device=device)

    def _hpsv3_score(prompts, completions, **kwargs):
        if not all(torch.is_tensor(img) for img in completions):
            raise ValueError("All inputs must be torch.Tensor images.")

        # handles tmp save paths
        image_paths = []
        tmp_files = []
        for i, img in enumerate(completions):
            pil_img = to_pil_image(torch.clamp((img + 1) / 2, 0, 1))  # scale [-1,1] -> [0,1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            pil_img.save(tmp.name)
            image_paths.append(tmp.name)
            tmp_files.append(tmp)

        # Run inference
        with torch.no_grad():
            rewards = inferencer.reward(image_paths, prompts)
        scores = [reward[0].item() for reward in rewards]

        #remove tmp files
        for tmp in tmp_files:
            try:
                os.unlink(tmp.name)
            except FileNotFoundError:
                pass

        return scores

    return _hpsv3_score

def geneval_score(device):
    """
    Submits images to GenEval and computes a reward.
    Currently, this score seems to be sparse.
    Not a good choice for RL reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64 # you need to sync this with the global design. FIXME
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _geneval_score(completions, prompts, metadata, only_strict=True, **kwargs):
        del prompts
        if isinstance(completions, torch.Tensor):
            completions = torch.clamp((completions + 1) / 2, 0, 1)
            completions = (completions * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            completions = completions.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(completions, np.ceil(len(completions) / batch_size))
        metadatas_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120, proxies={"http": None, "https": None})
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return torch.tensor(all_strict_rewards)

    return _geneval_score

def hpsv3_remote(device):
    """Submits images to hpsv3 server and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64 # you need to sync this with the global design. FIXME
    url = "http://127.0.0.1:18087"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _hpsv3_score(completions, prompts, **kwargs):

        if isinstance(completions, torch.Tensor):
            completions = torch.clamp((completions + 1) / 2, 0, 1)
            completions = (completions * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            completions = completions.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(completions, np.ceil(len(completions) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))
        all_scores = []
        for image_batch, prompts_batched in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompts_batched,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120, proxies={"http": None, "https": None})
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
        # scale to ~[0,2]
        return torch.tensor(all_scores)/5

    return _hpsv3_score

def unifiedreward_score(device):
    """Submits images to UnifiedReward and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18088"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _UniRwd(completions, prompts, **kwargs):
        if isinstance(completions, torch.Tensor):
            completions = torch.clamp((completions + 1) / 2, 0, 1)
            completions = (completions * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            completions = completions.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(completions, np.ceil(len(completions) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(
            url,
            data=data_bytes,
            timeout=60,
            proxies={"http": None, "https": None}
            )
            # print("response: ", response)
            # print("response: ", response.content)
            response_data = pickle.loads(response.content)
            # print(response_data["scores"])

            all_scores += response_data["scores"]

        return torch.tensor(all_scores)/5

    return _UniRwd


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[0] if len(matches) >= 1 else ""
    return extracted_answer

def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
import subprocess
import json

def evaluate_code(code, test_cases):
    passed = 0
    total = len(test_cases)
    exec_timeout = 5

    for case in test_cases:
        process = subprocess.run(
            ["python3", "-c", f"{{code}}\\n{{case}}"],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

        if process.returncode != 0:  # Error in execution
            continue

        # If we get here, the assertion passed (no error)
        passed += 1

    success_rate = (passed / total)
    return success_rate

code_snippet = {code_literal}
test_cases = {test_cases_literal}
rate = evaluate_code(code_snippet, test_cases)
print("__PASS_RATE__", rate)
"""
    # 1. compute format rewards
    format_rewards = get_code_format_reward(language='python')(completions)

    # 2. collect scripts and their indices in the original array
    template = evaluation_script_template
    scripts = []
    valid_indices = []
    for i, (reward, completion) in enumerate(zip(format_rewards, completions)):
        if reward < 1:
            continue
        code = extract_code(completion[-1]["content"])
        tc = kwargs["test_cases"][i]
        scripts.append(
            template.format(
                code_literal=repr(code),
                test_cases_literal=repr(tc),
            )
        )
        valid_indices.append(i)

    # 3. execute scripts in parallel
    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )
    results = execution_provider.execute_scripts(scripts, ["python"] * len(scripts))

    # 4. fill results into a list of the same length as completions, and keep None for reward=0
    final_results = [0.0] * len(completions)
    for idx, res in zip(valid_indices, results):
        final_results[idx] = 2 * res

    return final_results



def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    import ast
    pattern = re.compile(
        rf"^"
        r"(?:(?!```)[\s\S])*?"
        rf"```{language}\n"    # match ```language\n
        r"(?:(?!```)[\s\S])*?"         # match any character, but not ```
        rf"```\n$",                     # match ``` and end of string
        re.DOTALL
    )

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content in completion_contents:
            # # First check if the format matches
            # import pdb; pdb.set_trace();
            # format_match = pattern.fullmatch(content)
            # if not format_match:
            #     rewards.append(0.0)
            #     continue
                
            # Extract code from between code blocks
            code_blocks = re.findall(rf"```{language}\n(.*?)```", content, re.DOTALL)
            if not code_blocks:
                rewards.append(0.0)
                continue
                
            # Get the first code block (in case there are multiple)
            code = code_blocks[0].strip()
            
            # Check syntax if it's Python code
            if language == "python":
                try:
                    ast.parse(code)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
                rewards.append(0.5 if syntax_valid else 0.25) #  grammar error get partial reward
            else:
                # For other languages, just check format for now
                rewards.append(0.5)
                
        return rewards

    return code_format_reward