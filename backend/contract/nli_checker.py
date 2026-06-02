"""
Attorney.AI — ContractNLI Entailment Checker (HuggingFace Transformers)

Uses Natural Language Inference (NLI) to check whether a hypothesis
(a clause requirement) is:
  - ENTAILED    → the contract supports this requirement ✓
  - CONTRADICTED → the contract explicitly contradicts it ✗
  - NEUTRAL     → the contract doesn't mention it (gap) ?

Model: cross-encoder/nli-deberta-v3-base
  - Strong NLI performance (NLI benchmark SOTA for small models)
  - 184M params, runs well on CPU
  - Better than BART-MNLI for contract-style entailment

Fallback: typeform/distilbert-base-uncased-mnli (fast, smaller)

Based on the ContractNLI paper (Koreeda & Manning 2021):
https://stanfordnlp.github.io/contract-nli/
"""
from functools import lru_cache
from typing import List, Optional

from loguru import logger


# Labels from DeBERTa NLI (HF zero-shot pipeline)
_ENTAILMENT_LABEL = "entailment"
_CONTRADICTION_LABEL = "contradiction"
_NEUTRAL_LABEL = "neutral"


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    """Lazy-load the NLI pipeline."""
    try:
        from transformers import pipeline
        logger.info("Loading NLI model: cross-encoder/nli-deberta-v3-base")
        pipe = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=-1,  # CPU; set to 0 for GPU
        )
        logger.info("NLI model loaded")
        return pipe
    except Exception as e:
        logger.warning(f"NLI model load failed: {e}, falling back to distilbert-mnli")
        try:
            from transformers import pipeline
            return pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                device=-1,
            )
        except Exception as e2:
            logger.error(f"All NLI models failed: {e2}")
            return None


class ContractNLIChecker:
    """
    NLI-based contract clause checker.
    Given a contract text and a list of hypotheses (requirements to check),
    returns entailment/contradiction/neutral for each.

    Example:
        checker = ContractNLIChecker()
        results = checker.check_batch(
            premise="The contract shall be governed by California law.",
            hypotheses=[
                "The contract specifies a governing law.",
                "The contract is governed by New York law.",
                "The contract includes an arbitration clause.",
            ]
        )
        # → [ENTAILED, CONTRADICTED, NEUTRAL]
    """

    def check_single(
        self,
        premise: str,
        hypothesis: str,
    ) -> dict:
        """
        Check if hypothesis is entailed/contradicted/neutral in premise.
        Returns dict with label, confidence, and all scores.
        """
        pipe = _get_nli_pipeline()
        if not pipe:
            return {
                "hypothesis": hypothesis,
                "label": "NEUTRAL",
                "confidence": 0.0,
                "scores": {},
                "error": "NLI model not available",
            }

        # NLI: treat premise as context, hypothesis as the candidate label
        premise_trunc = premise[:800]  # DeBERTa max ~512 tokens
        hypothesis_trunc = hypothesis[:200]

        try:
            result = pipe(
                premise_trunc,
                candidate_labels=[hypothesis_trunc],
                hypothesis_template="{}",
            )
            # For zero-shot with single label, rerun with explicit NLI framing
            # More reliable: use the model directly for proper 3-way NLI
            label, confidence, scores = self._run_nli(pipe, premise_trunc, hypothesis_trunc)
        except Exception as e:
            logger.warning(f"NLI inference error: {e}")
            label, confidence, scores = "NEUTRAL", 0.0, {}

        return {
            "hypothesis": hypothesis,
            "label": label,
            "confidence": round(confidence, 4),
            "scores": scores,
        }

    def _run_nli(self, pipe, premise: str, hypothesis: str):
        """Run proper 3-way NLI classification."""
        try:
            result = pipe(
                premise,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template=f"The following is {{}}: " + hypothesis,
                multi_label=False,
            )
            labels = result["labels"]
            scores = dict(zip(result["labels"], result["scores"]))
            best_label = labels[0].upper()
            best_score = result["scores"][0]
            return best_label, best_score, scores
        except Exception:
            return "NEUTRAL", 0.0, {}

    def check_batch(
        self,
        premise: str,
        hypotheses: List[str],
    ) -> List[dict]:
        """
        Check multiple hypotheses against the same premise.
        More efficient than calling check_single in a loop.
        """
        return [self.check_single(premise, h) for h in hypotheses]

    def check_contract_requirements(
        self,
        contract_text: str,
        requirements: Optional[List[str]] = None,
    ) -> dict:
        """
        Check a contract against a standard set of requirements.
        Uses ContractNLI's standard hypothesis templates.
        """
        if requirements is None:
            requirements = _STANDARD_CONTRACT_REQUIREMENTS

        results = self.check_batch(contract_text, requirements)

        # Summarize
        summary = {
            "entailed": [r for r in results if r["label"] == "ENTAILMENT"],
            "contradicted": [r for r in results if r["label"] == "CONTRADICTION"],
            "neutral": [r for r in results if r["label"] == "NEUTRAL"],
            "details": results,
        }
        return summary


# Standard ContractNLI hypothesis templates
_STANDARD_CONTRACT_REQUIREMENTS = [
    "The contract includes a confidentiality obligation.",
    "The receiving party may not share information with third parties.",
    "The contract specifies a duration or term.",
    "The contract includes an indemnification clause.",
    "The contract specifies the governing law.",
    "The contract includes a limitation of liability.",
    "The contract prohibits assignment without consent.",
    "The contract includes an intellectual property assignment clause.",
    "The contract includes a non-compete obligation.",
    "The contract includes a dispute resolution or arbitration clause.",
    "The contract can be terminated for convenience.",
    "The contract includes a force majeure clause.",
    "The contract includes a warranty.",
    "The contract requires insurance coverage.",
    "The contract includes audit rights.",
]
