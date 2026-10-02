# Linear search in the label-flip budget

This derivation sharpens the existing extreme-candidate reduction for fixed one-sided WRS and pooled Student tests. It does not change the hypothesis families, BH procedure, label-flip uncertainty set, or reference construction.

## 1. Fixed net changes

Let the original groups have sizes m and n−m. Write i for removals from A and j for additions from B. Allowed changes satisfy i,j≥0 and i+j≤k. A net change d=j−i produces first-group size r=m+d. There are 2k+1 possible net changes, −k≤d≤k.

For a given d, j=i+d and the complete feasible interval is

\[
L_d=\max(0,-d),\qquad U_d=\left\lfloor\frac{k-d}{2}\right\rfloor,
\qquad L_d\le i\le U_d.
\]

We assume k≤min(m,n−m), so the original group-capacity constraints are redundant. For WRS the manuscript's stronger restriction k<min(m,n−m) leaves both resulting groups nonempty. For pooled Student, k≤min(m,n−m)−2 leaves at least two subjects per group. These restrictions make every d admissible. Budgets that would empty a group are rejected by the implementation rather than silently truncating the uncertainty set.

For WRS, the values below are pooled average ranks. For pooled Student they are the centered observed values. Sorting is performed separately inside each original group, retaining multiplicities of tied observations.

## 2. Convexity within each net change

For the minimum sum, put the A values in descending order a₁≥⋯≥a_m and the B values in ascending order b₁≤⋯≤b_(n−m). Let A_i and B_j be their prefix sums, with A₀=B₀=0, and let S₀ be the original first-group sum. The existing extreme-candidate argument gives

\[
F_d(i)=S_0-A_i+B_{i+d}.
\]

Every labeling at net change d and removal count i has a sum at least F_d(i), and this value is attained by flipping exactly those extreme original subjects. The forward difference is

\[
\Delta_d(i)=F_d(i+1)-F_d(i)=b_{i+d+1}-a_{i+1}.
\]

It is nondecreasing in i. Hence F_d is discretely convex. Its smallest minimizer q_d is the first feasible i with Δ_d(i)≥0, or U_d if every feasible increment is negative. Equivalently, at q_d the preceding increment is strictly negative unless q_d=L_d, and the following increment is nonnegative unless q_d=U_d. This convention chooses the fewest total flips among tied minimum-sum candidates at the same d.

The maximum-sum problem is identical after negating all observations: remove ascending A values and add descending B values. Its increments are nonincreasing, and the smallest maximizing i is obtained by reversing the inequality tests below.

## 3. A constrained sweep with a monotone pointer

Sweep d from −k to k. Both L_d and U_d are nonincreasing. On the common domain,

\[
\Delta_{d+1}(i)=b_{i+d+2}-a_{i+1}\ge\Delta_d(i).
\]

Suppose q_d is the smallest minimizer at the previous net change. Initialize the next pointer to c=min(q_d,U_(d+1)). This c is feasible: q_d≥L_d≥L_(d+1), and U_(d+1)≥L_(d+1).

If c=U_(d+1), it has no feasible increment to its right. Otherwise c=q_d<U_(d+1)≤U_d, so the old optimality condition and the displayed inequality imply Δ_(d+1)(c)≥0. Thus there is always a minimizer at or to the left of c. While c>L_(d+1) and the preceding increment Δ_(d+1)(c−1)≥0, decrease c by one. Each decrease preserves the nonnegative-increment condition on the right. On stopping, the left increment is strictly negative or the lower boundary is reached, so c is exactly the smallest constrained minimizer.

At d=−k, the feasible interval is the singleton {k}; initialize q=k. At d=k it is the singleton {0}. The pointer only decreases throughout the sweep and makes exactly k unit decreases in total, including decreases imposed by the shrinking upper boundary. The 2k+1 net changes plus at most k decreases establish O(k) work per endpoint after prefix sums are available. Finding O(k) candidates alone would not establish this claim; the monotone-pointer proof supplies the search bound as well.

### Pseudocode: minimum-sum candidates

```text
Input: descending A values a, ascending B values b, budget k, original sum S0
Build prefix sums A[0..k], B[0..k]
q ← k
for d = −k, …, k:
    L ← max(0, −d)
    U ← floor((k − d)/2)
    q ← min(q, U)
    while q > L and b[q+d] ≥ a[q]:       # one-based value indices
        q ← q − 1
    emit (r=m+d, S=S0−A[q]+B[q+d])
```

For maxima, sort A ascending and B descending and use b[q+d]≤a[q]. Equalities deliberately move the pointer left; ties neither invalidate convexity nor increase the work bound. The loop guard ensures both value indices are at least one; the budget constraint keeps them at most k.

The radius constraint is **at most k**, including the original labeling. It is not restricted to the shell i+j=k. In particular the d=0 optimizer can use any feasible number of opposite-direction flips, including zero. The original labeling need not be evaluated separately once its net-change class has been optimized.

## 4. From sums to p-value endpoints

At fixed r, the WRS null distribution is fixed by the pooled ranks, tie pattern and r. Exact one-sided permutation p-values are monotone in the rank sum. The tie-corrected normal approximation used in the experiments has the same monotonicity. Thus the two extreme sums at each d suffice for both fixed one-sided endpoints, including ties. The reduction to O(k) null-distribution evaluations applies to exact permutation p-values as well; constructing or querying those distributions has an additional cost that is not hidden in the arithmetic bound.

For nonconstant pooled data, write Q=Σ(x−mean(x))² and S for the first group's centered sum. At fixed r,

\[
t_r(S)=\sqrt{\frac{n(n-2)}{r(n-r)}}\,
\frac{S}{\sqrt{Q-\frac{nS^2}{r(n-r)}}}
\]

is increasing wherever the denominator is positive. The signed infinite-t limits at a nonconstant zero-within-group-variance partition preserve monotonicity. Therefore the same sweep gives both pooled one-sided endpoints. Entirely constant or unavailable features use the existing separate p=1 convention. Welch's statistic is not covered.

Each endpoint requires at most 2k+1 net-change candidates, rather than (k+1)(k+2)/2 count-pair candidates. Computing both sum extrema uses at most 4k+2 evaluations; duplicates can be reused. With the fixed normal and Student tail maps, the implementation takes extrema of the standardized statistics first and evaluates the two tails only at their final extrema.

## 5. Complexity and implementation qualifications

- One feature: O(n log n+k) including ordinary sorting, with O(k) extra prefix/pointer storage after the sorted input is available. WRS ranking and pooled centering fit within the preprocessing term.
- N features: O(N n log n+Nk). Applying BH once adds O(N log N), for ERDEG. The joint family has 2N hypotheses and the same asymptotic order.
- Exact RDEG is unchanged: BH under common labelings still involves a separate combinatorial calculation.
- Computing bounds separately for every budget 0,…,K costs O(K²) after reusable sorting. This result is for one specified budget, not a simultaneous linear-time algorithm for the entire budget curve.
- The proof is in exact/unit-cost arithmetic. Floating-point centered moments inherit the conditioning limitations of the existing pooled implementation. This is distinct from the count-pair search proof.
- In a batched implementation, repeatedly scanning all N features during every inner pointer move can lose the O(Nk) guarantee. The supplied implementation restricts subsequent inner iterations to still-moving feature indices. It records at most k total decrements and at most 3k+1 comparison operations per feature and endpoint.
- The proof establishes exact optimization of the specified p-value rule; it does not establish statistical calibration, a measured speedup at every budget, or novelty relative to all prior work.
