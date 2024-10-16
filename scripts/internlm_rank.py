import numpy as np
from tqdm import tqdm


def rank_wrapper(
    inputs: List[str],
    candidates: List[List[str]],
    model,
    tokenizer,
    instructions: List[str] = None,
    return_scores: bool = False,
    batch_size: int = 8,
    disable_tqdm: bool = False,
    **rank_kwargs,
):
    """
    Args:
        inputs (List[str]): List of input texts.
        candidates (List[List[str]]): List of list of candidate texts.
        model: The AutoModel instance.
        tokenizer: The AutoTokenizer instance.
        instructions (List[str], optional): List of instructions.
        return_scores (bool, optional): If True, returns scores instead of ranks.
        batch_size (int, optional): Batch size for ranking.
        disable_tqdm (bool, optional): If True, disables the progress bar.
    Returns:
        List[List[int]] or List[List[float]]: Ranks or scores of candidates for each input.
    """
    # Validations
    assert len(inputs) == len(
        candidates
    ), "Number of inputs and candidates must be the same"
    assert all(
        [len(c) > 0 for c in candidates]
    ), "Each input must have at least one candidate"
    assert all(
        [len(c) == len(candidates[0]) for c in candidates]
    ), "Number of candidates for each input must be the same"

    n_inputs = len(inputs)
    n_candidates = len(candidates[0])

    # Create chats for each input and its candidates
    all_chats = []
    for idx, (input_text, candidate_texts) in enumerate(zip(inputs, candidates)):
        if instructions is not None and len(instructions) > idx:
            instruction = instructions[idx]
            input_text = f"{instruction}\n{input_text}"
        for candidate_text in candidate_texts:
            chat = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": candidate_text},
            ]
            all_chats.append(chat)

    # Compute scores in batches
    scores = []
    for i in tqdm(
        range(0, len(all_chats), batch_size),
        desc="Ranking candidates",
        disable=disable_tqdm,
    ):
        batch_chats = all_chats[i : i + batch_size]
        batch_scores = model.get_scores(tokenizer, batch_chats)
        scores.extend(batch_scores)

    # Reshape scores into a 2D list [n_inputs x n_candidates]
    scores_matrix = []
    index = 0
    for _ in range(n_inputs):
        input_scores = scores[index : index + n_candidates]
        scores_matrix.append(input_scores)
        index += n_candidates

    if return_scores:
        return scores_matrix
    else:
        # Convert scores to ranks (lower rank is better)
        ranks_matrix = []
        for input_scores in scores_matrix:
            sorted_indices = np.argsort(-np.array(input_scores))
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(input_scores) + 1)
            ranks = ranks.tolist()
            ranks_matrix.append(ranks)
        return ranks_matrix
