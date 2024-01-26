import argparse
import sys
import os
import shutil
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 
from model_merging_methods.merging_methods import MergingMethod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--m1", type=str, default="")
    parser.add_argument("--m2", type=str, default="")
    parser.add_argument("--llama-path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--p", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.llama_path, use_fast=False)
    llama1 = AutoModelForCausalLM.from_pretrained(args.llama_path)
    llama2 = AutoModelForCausalLM.from_pretrained(args.llama_path)

    pretrained_model = AutoModelForCausalLM.from_pretrained(args.llama_path)

    model1 = PeftModel.from_pretrained(llama1, args.m1).merge_and_unload()
    model2 = PeftModel.from_pretrained(llama2, args.m2).merge_and_unload()

    merging_method = MergingMethod(merging_method_name="mask_merging")

    merged_model = pretrained_model
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=[model1, model2],
                                                   exclude_param_names_regex=[],
                                                   trainers=[None, None],
                                                   scaling_coefficient=None,
                                                   nums_fisher_examples=None, 
                                                   fisher_scaling_coefficients=None,
                                                   normalize_fisher_weight=None,
                                                   minimal_fisher_weight=None,
                                                   nums_regmean_examples=None,
                                                   reduce_non_diagonal_ratio=None,
                                                   param_value_mask_rate=None,
                                                   weight_format="delta_weight",
                                                   weight_mask_rates=[0.9,0.9],
                                                   use_weight_rescale=True,
                                                   mask_strategy="random",
                                                   mask_apply_method="average_merging",
                                                   models_use_deepcopy=False)


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    tokenizer.save_pretrained(args.save_path)
    merged_model.save_pretrained(args.save_path)
if __name__ == "__main__":
    main()