import sys
import time
from evaluation import evaluate_lang_0
import logging
import os
import openai
import csv
from utils import save_to_file
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import pickle

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def gpt3_complete(prompt, max_tokens, model_name, temp=0, num_log_probs=None, echo=False, n=1):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(model=model_name, prompt=prompt, max_tokens=max_tokens, temperature=temp, n=n,
                                                logprobs=num_log_probs, echo=echo, stop='\n')
            received = True
        except openai.error.InvalidRequestError as e:
            print("API error: ", e)
            if "maximum context length" in e:
                print("returning None bc the prompt was too long")
                return None
        except Exception as e:
            print("API error: ", e)
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def chatgpt_complete(message, l, model_name, temp=0, n=1):
    # call ChatGPT API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(model=model_name, messages=message, max_tokens=l, temperature=temp, n=n, stop='\n')
            received = True
        except openai.error.InvalidRequestError as e:
            print("API error: ", e)
            if "maximum context length" in e:
                print("returning None bc the prompt was too long")
                return None
        except Exception as e:
            print("API error: ", e)
    return response