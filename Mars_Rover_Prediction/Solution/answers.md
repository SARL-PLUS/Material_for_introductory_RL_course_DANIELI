How much does the verdict “TD (0) is better than first-visit MC” depend on the constant step-size α we happened to test?

Fact	TD (0)	First-Visit MC
Update target noise	Low (bootstraps on a single next-state sample)	High (full-return variance grows with episode length)
Bias when α is constant	Non-zero, proportional to α	Non-zero, proportional to α
Best usable α (rule-of-thumb)	0.05 – 0.3 on small tabular tasks	0.005 – 0.1 (needs ≈10× smaller)
What happens if α is too large?	Slow oscillations or divergence only at the extreme	Highly unstable, huge variance



⸻

1 Sweeping α over a wide range will not flip the early-learning verdict

If you redraw the learning-curve grid for, say,
\alpha\in\{0.001,\;0.005,\;0.01,\;0.02,\;0.05,\;0.1,\;0.2,\;0.4\},
three robust patterns appear:
	1.	For any α that keeps both methods stable (roughly ≤ 0.3 for TD, ≤ 0.1 for MC), TD’s RMSE is lower for the first few-dozen episodes.
The bootstrap target just has less noise, so TD moves in the right direction more often.
	2.	MC becomes competitive only when α is so small that early learning is extremely slow (e.g. α ≤ 0.01).
In that regime TD is also almost static, so whoever wins depends on minute differences in residual bias rather than on useful learning.
	3.	At very large α MC blows up first (wild swings or divergence) while TD still degrades gracefully.

Hence, scanning a wide constant-α grid does not reverse the practical conclusion:
“If you have a limited episode budget, TD (0) is the safer, faster choice.”

⸻

2 Is there a magic fixed α where MC suddenly shines?

Not really.  For a given constant α both methods settle (in expectation) at an RMSE plateau
\text{MSE}\infty \;\propto\; \frac{\sigma^2{\text{target}}\;\alpha}{2-\alpha\,\lambda_{\max}},
where \sigma^2_{\text{target}} is the variance of the update target and \lambda_{\max}<1 depends on the Markov chain under the policy.
Because \sigma^2_{\text{target}}^{\text{MC}}\gg\sigma^2_{\text{target}}^{\text{TD}}, MC needs an α roughly an order of magnitude smaller just to reach the same asymptotic error.  That smaller α slows its learning proportionally.  There is therefore no single α that makes MC both (i) stable and (ii) faster than TD over any realistic horizon.

⸻

3 What would change the story?
	•	Diminishing step-sizes (αₜ ↓ 0, e.g. 1/t) remove steady-state bias; then both methods converge to the true V. TD still wins on sample efficiency, but MC eventually catches up.
	•	Batch/least-squares variants (LSTD, LS-MC) solve for V in closed form, eliminating the α problem altogether.
	•	Off-policy settings can favour MC because its target is independent of the behaviour policy, whereas TD must correct with importance sampling.

⸻

Bottom line

Varying α over a wide range does not overturn the observation that TD(0) learns a good value function in far fewer episodes than first-visit Monte-Carlo.
There is no “sweet-spot” constant α that lets MC simultaneously keep variance under control and learn quickly; picking α small enough for stability slows MC so much that TD remains preferable for any finite data budget.