sqlite-utils query data/results/eval_results.db "SELECT is_correct, COUNT(*) FROM df GROUP BY is_correct;"
sqlite-utils query data/results/eval_context_results.db "SELECT is_correct, COUNT(*) FROM df GROUP BY is_correct;"
