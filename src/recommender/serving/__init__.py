"""Serving layer: precomputed top-N publication and the Lambda handler."""

from recommender.serving.handler import lambda_handler, parse_request
from recommender.serving.topn import build_topn_payloads

__all__ = ["build_topn_payloads", "lambda_handler", "parse_request"]
