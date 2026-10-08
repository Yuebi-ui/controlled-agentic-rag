"""Reproducible, annotation-aware evaluation for Controlled Agentic RAG.

Functions calculate metrics from supplied labels / traces; they do NOT produce
or infer semantic ground truth from unannotated model answers.
"""
from .runner import evaluate_suite

__all__ = ['evaluate_suite']
