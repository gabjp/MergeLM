import argparse
import sys
import logging
import os
import time
from vllm import LLM, SamplingParams

from inference_llms_instruct_math_code import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for direct inference merged LLMs")
    parser.add_argument("--model-path", type=str)
    parser.add_argument('--start_index', type=int, default=0)
    
    args = parser.parse_args()


    llm = LLM(model=args.model_path, tensor_parallel_size=1)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

        
    save_gen_results_folder = f"./completions"
    test_human_eval(llm=llm, args=args, logger=logger,
                    save_model_path=None, save_gen_results_folder=save_gen_results_folder)


    logger.info(f"inference of merging method {args.merging_method_name} is completed")

    sys.exit()
