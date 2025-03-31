from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List

from mediqa.config.core import ReaderConfig

SYS_PROMPT = """
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.
"""
USR_PROMPT = """
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
"""


class Reader:
    def __init__(self, config: ReaderConfig):
        self.config = config
        
        bnb_config = None
        if config.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            return_full_text=False,
            max_new_tokens=config.max_new_tokens,
        )

        self.RAG_prompt_template = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": USR_PROMPT},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, questions_batch: List[str], retrieval_batch: List[List[str]]) -> List[str]:
        # Construct the batched context
        assert len(questions_batch) == len(retrieval_batch), "Batch size is not consistent for questions and context"
        final_prompt_batch = []
        for question, retrieval in zip(questions_batch, retrieval_batch):
            retrieval_text = retrieval["documents"][0]
            context = "\nExtracted documents:\n"
            context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieval_text)])

            final_prompt = self.RAG_prompt_template.format(question=question, context=context)
            final_prompt_batch.append(final_prompt)

        answers = self.pipe(final_prompt_batch)

        return answers
