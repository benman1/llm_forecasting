"""Retrieval and reasoning.

This runs the system on a single question.
"""

from typing import List, Dict

import ensemble
import ranking
import summarize
from prompts.prompt_collection import PROMPT_DICT
from utils.visualize_utils import visualize_all, visualize_all_ensemble

RETRIEVAL_CONFIG = {
    "NUM_SEARCH_QUERY_KEYWORDS": 3,
    "MAX_WORDS_NEWSCATCHER": 5,
    "MAX_WORDS_GNEWS": 8,
    "SEARCH_QUERY_MODEL_NAME": "gpt-4-1106-preview",
    "SEARCH_QUERY_TEMPERATURE": 0.0,
    "SEARCH_QUERY_PROMPT_TEMPLATES": [
        PROMPT_DICT["search_query"]["0"],
        PROMPT_DICT["search_query"]["1"],
    ],
    "NUM_ARTICLES_PER_QUERY": 5,
    "SUMMARIZATION_MODEL_NAME": "gpt-3.5-turbo-1106",
    "SUMMARIZATION_TEMPERATURE": 0.2,
    "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
    "PRE_FILTER_WITH_EMBEDDING": True,
    "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
    "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
    "RANKING_TEMPERATURE": 0.0,
    "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
    "RANKING_RELEVANCE_THRESHOLD": 4,
    "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
    "SORT_BY": "date",
    "RANKING_METHOD": "llm-rating",
    "RANKING_METHOD_LLM": "title_250_tokens",
    "NUM_SUMMARIES_THRESHOLD": 20,
    "EXTRACT_BACKGROUND_URLS": True,
}


REASONING_CONFIG = {
    "BASE_REASONING_MODEL_NAMES": ["gpt-4-1106-preview", "gpt-4-1106-preview"],
    "BASE_REASONING_TEMPERATURE": 1.0,
    "BASE_REASONING_PROMPT_TEMPLATES": [
        [
            PROMPT_DICT["binary"]["scratch_pad"]["1"],
            PROMPT_DICT["binary"]["scratch_pad"]["2"],
        ],
        [
            PROMPT_DICT["binary"]["scratch_pad"]["new_3"],
            PROMPT_DICT["binary"]["scratch_pad"]["new_6"],
        ],
    ],
    "ALIGNMENT_MODEL_NAME": "gpt-3.5-turbo-1106",
    "ALIGNMENT_TEMPERATURE": 0,
    "ALIGNMENT_PROMPT": PROMPT_DICT["alignment"]["0"],
    "AGGREGATION_METHOD": "meta",
    "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
    "AGGREGATION_TEMPERATURE": 0.2,
    "AGGREGATION_MODEL_NAME": "gpt-4",
    "AGGREGATION_WEIGTHTS": None,
}


def retrieve_and_reason(
    question: str,
    background_info: str,
    resolution_criteria: str,
    question_dates: List[str],
    retrieval_dates: List[str, str],
    urls_in_background: List[str],
    answer: str,
    raw_data=None,
    do_visualize: bool = False,
) -> Dict:
    """Run the full prediction pipeline on a single question with background.

    Returns a dictionary of predictions.
    """
    (
        ranked_articles,
        all_articles,
        search_queries_list_gnews,
        search_queries_list_nc,
    ) = await ranking.retrieve_summarize_and_rank_articles(
        question,
        background_info,
        resolution_criteria,
        retrieval_dates,
        urls=urls_in_background,
        config=RETRIEVAL_CONFIG,
        return_intermediates=True,
    )
    print("Question:", question)
    print("Background:", background_info)
    (
        ranked_articles,
        all_articles,
        search_queries_list_gnews,
        search_queries_list_nc,
    ) = await ranking.retrieve_summarize_and_rank_articles(
        question,
        background_info,
        resolution_criteria,
        retrieval_dates,
        urls=urls_in_background,
        config=RETRIEVAL_CONFIG,
        return_intermediates=True,
    )
    all_summaries = summarize.concat_summaries(
        ranked_articles[: RETRIEVAL_CONFIG["NUM_SUMMARIES_THRESHOLD"]]
    )
    today_to_close_date = (retrieval_dates[1], question_dates[1])
    ensemble_dict = await ensemble.meta_reason(
        question=question,
        background_info=background_info,
        resolution_criteria=resolution_criteria,
        today_to_close_date_range=today_to_close_date,
        retrieved_info=all_summaries,
        reasoning_prompt_templates=REASONING_CONFIG["BASE_REASONING_PROMPT_TEMPLATES"],
        base_model_names=REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
        base_temperature=REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
        aggregation_method=REASONING_CONFIG["AGGREGATION_METHOD"],
        answer_type="probability",
        weights=REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
        meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
        meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
        meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
    )
    if do_visualize:
        base_brier_scores = rate_answers(ensemble_dict, answer=answer)
        base_html = visualize_all(
            question_data=raw_data[0],
            retrieval_dates=retrieval_dates,
            search_queries_gnews=search_queries_list_gnews,
            search_queries_nc=search_queries_list_nc,
            all_articles=all_articles,
            ranked_articles=ranked_articles,
            all_summaries=all_summaries,
            model_names=REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
            base_reasoning_prompt_templates=REASONING_CONFIG[
                "BASE_REASONING_PROMPT_TEMPLATES"
            ],
            base_reasoning_full_prompts=ensemble_dict["base_reasoning_full_prompts"],
            base_reasonings=ensemble_dict["base_reasonings"],
            base_predictions=ensemble_dict["base_predictions"],
            base_brier_scores=base_brier_scores,
        )
        meta_html = visualize_all_ensemble(
            question_data=raw_data[0],
            ranked_articles=ranked_articles,
            all_articles=all_articles,
            search_queries_gnews=search_queries_list_gnews,
            search_queries_nc=search_queries_list_nc,
            retrieval_dates=retrieval_dates,
            meta_reasoning=ensemble_dict["meta_reasoning"],
            meta_full_prompt=ensemble_dict["meta_prompt"],
            meta_prediction=ensemble_dict["meta_prediction"],
        )
        base_file_path = "sample_q_base_output.html"
        meta_file_path = "sample_q_meta_output.html"

        with (
            open(base_file_path, "w") as base_file,
            open(meta_file_path, "w") as meta_file,
        ):
            base_file.write(base_html)
            meta_file.write(meta_html)
    return ensemble_dict


def rate_answers(ensemble_dict: Dict, answer: str) -> List[List[float]]:
    """Compute brier score (base_predictions is a list of lists of probabilities.

    You can take the aggregate decision as well: ensemble_dict["meta_prediction"].
    """
    base_brier_scores = []
    # For each sublist (corresponding to a base model name)
    for base_predictions in ensemble_dict["base_predictions"]:
        base_brier_scores.append(
            [(base_prediction - answer) ** 2 for base_prediction in base_predictions]
        )

    print(base_brier_scores)
    return base_brier_scores
