"""Pluggable embeddings + LLM: openai | local (free) | fake (offline tests)."""
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from . import config


def get_embeddings() -> Embeddings:
    backend = config.resolve_backend("embed")
    if backend == "openai":  # paid API, best quality
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL)
    if backend == "fake":  # deterministic vectors, no downloads -> CI/smoke tests
        from langchain_core.embeddings.fake import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=384)
    # default: free local sentence-transformers model
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=config.LOCAL_EMBED_MODEL, model_kwargs={"device": "cpu"}
    )


def get_llm() -> BaseLanguageModel:
    backend = config.resolve_backend("llm")
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_CHAT_MODEL, temperature=0)
    if backend == "fake":  # canned reply, proves the wiring without a model
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        return GenericFakeChatModel(messages=iter(["[fake-llm] answer based on the retrieved context"]))
    # default: free local chat model running on CPU
    # transformers v5 removed text2text-generation -> use text-generation + chat wrapper
    from transformers import pipeline
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipe = pipeline(
        "text-generation", model=config.LOCAL_CHAT_MODEL, max_new_tokens=256
    )
    # pass the pipeline's own tokenizer so its chat template is found
    # return_full_text=False strips the echoed prompt from the output
    return ChatHuggingFace(
        llm=HuggingFacePipeline(
            pipeline=pipe, pipeline_kwargs={"return_full_text": False}
        ),
        tokenizer=pipe.tokenizer,
    )
