# Latent-Space Interpolation with Generative Flow Models

*Problem definition, assumptions, experimental evidence, and near-term research plan*

| **Project**  | Interpolation of video frames and anisotropic image slices using a pretrained 2D generative model |
|--------------|---------------------------------------------------------------------------------------------------|
| **Status**   | Working research formulation; experimental assumptions remain under active evaluation             |
| **Prepared** | July 2026                                                                                         |

> **Working thesis**
>
> A deterministic generative transport can provide a useful coordinate system for interpolation, but prior typicality alone does not guarantee that a simple spherical path in noise space follows the encoded trajectory of a real temporal or spatial process. Success therefore depends jointly on the interpolation rule and on the geometry induced by model training.

## Purpose of this memo

This document consolidates the current problem statement, defines the main objects and terms, records the assumptions being made, summarizes what the experiments support or weaken, and lays out a research plan that does not initially rely on privileged access to simulator state or a ground-truth generative process.

## Scope boundary

- **Primary scope:** deterministic endpoint-conditioned interpolation using a pretrained 2D flow or diffusion model.

- **Target applications:** video frame interpolation and interpolation of sparsely sampled 2D slices from a 3D volume.

- **Current testbed:** synthetic sequences such as a bouncing ball, with dense frames available for evaluation.

- **Not assumed:** a reliable physical-state metric, simulator state labels, or a sequence-native generative model.

# 1. Formal problem statement

Let X denote the data space of individual 2D frames or slices, and let Z denote the latent/noise space of a generative transport model. We assume a trained deterministic ordinary differential equation defines a decoder G: Z → X and, under appropriate numerical treatment, an approximate inverse encoder E: X → Z.

Given two observed endpoint samples A and B, or a sequence of observed keyframes, the objective is to generate intermediate samples that satisfy two requirements:

- **Marginal realism:** each generated intermediate should conform to the learned data distribution.

- **Conditional validity:** each generated intermediate should be a plausible state between the surrounding observations at the requested temporal or spatial coordinate.

## For a two-endpoint interval with normalized coordinate s in [0, 1], the current model class is:

```text
z_A = E(A),    z_B = E(B)
z_hat(s) = f(z_A, z_B, s)
x_hat(s) = G(z_hat(s))
```

The first research question is to identify an interpolation function f that produces a good approximation to the unknown encoded trajectory. The second is to determine which model-training choices make such a simple f possible and stable.

> **Conditional formulation**
>
> The desired target is not merely a high-probability frame under p_data(x). It is closer to a conditional distribution p(x_s | x_0 = A, x_1 = B, s). A realistic frame at the wrong position or phase is not a valid intermediate.

## Research questions

- **RQ1 - Interpolator:** What simple curve family in Z best approximates encoded data trajectories between keyframes?

- **RQ2 - Representation:** Which training objectives and source-target couplings yield a latent map in which nearby data states remain nearby and trajectories remain smooth?

- **RQ3 - Identifiability:** Under what temporal spacing, dynamics, and image-formation conditions are the endpoints sufficient to determine a useful intermediate?

- **RQ4 - Generality:** Do conclusions transfer across rigid motion, periodic motion, deformation, occlusion, and slice-interpolation processes?

- **RQ5 - Evaluation:** Which latent and image-space diagnostics distinguish typicality, temporal consistency, conditional accuracy, and numerical stability?

# 2. Definitions and notation

| **Symbol or term**   | **Definition**                                                                                                                            |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| X                    | Data space of individual frames or image slices.                                                                                          |
| Z                    | Latent/noise space, nominally distributed as a high-dimensional standard Gaussian.                                                        |
| p_data               | Marginal distribution of individual data samples.                                                                                         |
| p_Z                  | Prior distribution in latent/noise space, typically N(0, I).                                                                              |
| G: Z → X           | Deterministic noise-to-data map induced by ODE sampling.                                                                                  |
| E: X → Z           | Approximate inverse data-to-noise map obtained by integrating the ODE in reverse.                                                         |
| x\*(s)               | A true continuous frame or slice trajectory indexed by physical time or slice coordinate s.                                               |
| z\*(s)               | The encoded trajectory E(x\*(s)). This can be observed on synthetic dense sequences even when the underlying simulator state is not used. |
| Keyframe             | An observed frame used as an interpolation constraint.                                                                                    |
| f                    | Latent interpolation rule, such as SLERP, SQUAD, or a future alternative.                                                                 |
| Typical set          | The high-probability shell of a high-dimensional Gaussian, concentrated near radius sqrt(dim Z).                                          |
| SLERP                | Great-circle interpolation on a fixed-radius hypersphere; preserves typical radius better than Euclidean linear interpolation.            |
| SQUAD                | A spherical spline through multiple latent keyframes that improves tangent continuity relative to piecewise SLERP.                        |
| Hybrid interpolation | A diagnostic method that combines an image-space interpolation with a latent interpolation at an intermediate diffusion/flow time.        |
| Conditional validity | Consistency with the ordered process between endpoints, not only with the marginal frame distribution.                                    |

# 3. Decomposition of the core assumptions

The initial hypothesis can be decomposed into distinct assumptions. This prevents evidence for one property - for example Gaussian typicality - from being interpreted as evidence for a different property such as temporal correctness.

| **Assumption**                               | **Statement**                                                                                                             | **Current status**                                                                                                                                                                                                                        |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A1. Reliable deterministic transport         | The learned ODE provides a stable, approximately one-to-one map between relevant regions of X and Z.                      | Supported for data → noise → data when endpoint handling and integration are controlled. Noise-space cycle accuracy is sensitive near t → 0, but this appears primarily numerical and is not the main cause of interpolation error. |
| A2. Prior typicality of encoded samples      | Encoded endpoints lie in, or near, the typical set of the Gaussian prior.                                                 | Supported by latent mean, standard deviation, and radius diagnostics. This justifies radius-preserving interpolation as a reasonable constraint.                                                                                          |
| A3. Continuous ordered process               | The data arise from a continuous trajectory in physical time or slice coordinate.                                         | Reasonable for the intended video and volumetric applications, although discontinuities, appearance events, and occlusion can violate simple smoothness.                                                                                  |
| A4. Endpoint sufficiency                     | The surrounding keyframes contain enough information to identify a useful intermediate.                                   | Likely valid for selected toy intervals and dense keyframes; not universally valid. This should be tested by varying keyframe spacing and generator class.                                                                                |
| A5. Latent locality                          | Nearby states in the ordered data process map to nearby points in Z.                                                      | Partially supported by smoother decoded sequences under SLERP/SQUAD. The degree of locality and its anisotropy remain to be characterized.                                                                                                |
| A6. Simple latent trajectory geometry        | The encoded trajectory can be approximated by a low-complexity curve such as SLERP or SQUAD.                              | SQUAD is a useful baseline but does not perfectly reproduce dense encoded trajectories. The endpoint-span and local-subspace residuals should directly test this assumption.                                                              |
| A7. Decoder regularity                       | Smooth latent curves decode to smooth and semantically consistent data trajectories.                                      | Smoothness is observed, but semantic correctness is incomplete. The decoder may strongly contract some latent directions near the image boundary.                                                                                         |
| A8. Appropriate latent metric                | The spherical metric used by SLERP/SQUAD is sufficiently aligned with the geometry relevant to data interpolation.        | Unvalidated. Spherical geometry enforces prior typicality, not necessarily the pullback geometry induced by G or the physical process.                                                                                                    |
| A9. Deterministic interpolation is preferred | For near-deterministic endpoints, stochastic innovations add uncertainty without useful endpoint-conditioned information. | Supported by bridge and data-consistency experiments: increasing stochasticity decreased fidelity and temporal consistency.                                                                                                               |
| A10. Keyframe density is adequate            | Observed keyframes are close enough for local interpolation assumptions to hold.                                          | Open. This is directly testable through keyframe-density sweeps and is likely process dependent.                                                                                                                                          |

## Typicality, trajectory geometry, and decoder geometry

These three properties should be evaluated separately:

- **Prior typicality:** Does the interpolated latent remain in a high-probability region of p_Z? SLERP directly addresses this.

- **Trajectory geometry:** Does the latent curve approximate z\*(s), the encoded trajectory of the ordered process? SLERP does not guarantee this.

- **Decoder geometry:** Does movement along the latent curve produce smooth, meaningful movement in X? This depends on the Jacobian and pullback geometry of G.

> **Important qualification**
>
> SLERP is the geodesic of a fixed-radius hypersphere, not the unconstrained Euclidean noise space and not necessarily the data manifold. Its theoretical motivation is preservation of Gaussian typical radius. Temporal correctness requires additional structure.

# 4. Current method and experimental interpretation

## Current latent interpolation pipeline

- **1. Encode:** map observed keyframes from image space into terminal latent/noise space using E.

- **2. Interpolate:** construct intermediate latent states using SLERP for two endpoints or SQUAD for multiple keyframes.

- **3. Decode:** integrate the deterministic ODE from each interpolated latent to image space.

- **4. Evaluate:** compare against dense synthetic ground-truth frames using image metrics, temporal metrics, and latent-trajectory diagnostics.

## What the experiments currently support

| **Experiment**              | **Observation**                                                                                                                                                        | **Interpretation**                                                                                                                                                    |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Round-trip evaluation       | Data → noise → data is accurate. Large noise-cycle error is concentrated extremely close to the image boundary and decreases strongly with integration resolution. | The learned transport is usable; the boundary issue is unlikely to dominate one-way interpolation results.                                                            |
| SLERP/SQUAD interpolation   | Smooth latent paths generate smoother data sequences than independent sampling. SQUAD is generally slightly better than piecewise SLERP.                               | The latent representation contains useful local trajectory structure, but the spherical spline is not an exact model of the encoded trajectory.                       |
| Stochastic bridge           | Adding endpoint-vanishing stochastic residuals increases diversity but decreases fidelity and temporal consistency as magnitude increases.                             | Unconditioned stochasticity contributes no information about the true intermediate and is unsuitable as a correction mechanism for nearly deterministic trajectories. |
| ISCS-style data consistency | Observed keyframes are exact, but missing frames remain effectively marginal samples because the temporal masking operator does not couple them to the endpoints.      | Noise correlation alone cannot create endpoint conditioning.                                                                                                          |
| Hybrid interpolation        | A small image-space contribution combined with a mostly latent state can modestly outperform pure latent interpolation at selected mix times.                          | The latent path provides realism and smoothness, while the image interpolation appears to supply a crude correspondence cue missing from the naive latent trajectory. |

## Interpretation of the hybrid result

The hybrid result should be treated as a diagnostic rather than the preferred final formulation. A 75-90% latent mixture performing best suggests that the SQUAD path is close to useful but misses some low-frequency correspondence or location information. The image interpolation supplies that information imperfectly, and the remaining ODE segment projects the mixed state toward the learned image distribution.

This supports the feasibility of latent interpolation while weakening the assumption that an independently trained flow will automatically align ordered data trajectories with spherical geodesics in Z.

## What is not currently supported

- That Gaussian typicality is sufficient for a valid conditional intermediate.

- That the true encoded trajectory remains in the endpoint plane required by two-point SLERP.

- That global optimality of the generative transport automatically preserves temporal neighborhoods.

- That stochastic exploration improves a nearly deterministic interpolation problem without endpoint-conditioned information.

- That a single toy generator is sufficient to establish generality.

# 5. Near-term research plan

The near-term plan prioritizes methods that use observed image sequences and their ordering, but do not require simulator state, a privileged physical metric, or direct supervision from an underlying generator process.

## 5.1 Characterize actual encoded trajectories

Dense synthetic sequences provide x\*(s), allowing z\*(s) = E(x\*(s)) to be measured without using the simulator state. This should be the first analysis because it directly tests whether a better closed-form interpolator is plausible or whether the representation itself must change.

- **SLERP/SQUAD error:** distance from z\*(s) to the corresponding latent interpolation, stratified by keyframe spacing and interpolation coordinate.

- **Endpoint-plane residual:** fraction of z\*(s) outside span(z_A, z_B), which lower-bounds the error of every two-endpoint great-circle method.

- **Local four-keyframe subspace residual:** analogous residual for the neighborhood used by SQUAD.

- **Radial versus angular motion:** decompose the true latent trajectory into radius changes and direction changes.

- **Speed, curvature, and acceleration:** measure whether physical time maps to approximately constant latent speed and where the curve bends sharply.

- **Tangent alignment:** compare the tangent of z\*(s) with the tangent of SLERP/SQUAD.

- **Density sweep:** repeat all measurements while varying keyframe spacing.

## 5.2 Latent acceleration loss

Add a weak sequence-order regularizer that promotes smooth latent trajectories for nearby observed frames. The preferred initial form should encourage low latent acceleration rather than force every interval to be a great circle.

Conceptually, for three nearby ordered frames:

```text
L_acc = || z_(k+1) - 2 z_k + z_(k-1) ||^2
```

For normalized spherical latents, a later refinement can compare tangent vectors using log maps and parallel transport. The loss should initially be local, lightly weighted, and evaluated for its effect on generation quality, prior statistics, and interpolation error.

## 5.3 Optimal-transport coupling experiments

Compare independent flow matching against minibatch optimal-transport couplings or related low-cost assignment strategies. The hypothesis is that a lower-cost and less crossing-prone source-target coupling may yield a smoother inverse map and better neighborhood preservation, even without explicit trajectory supervision.

- **Controls:** same model architecture, training budget, data augmentations, solver, and evaluation suite.

- **Primary outcomes:** latent trajectory curvature, SLERP/SQUAD error, endpoint-plane residual, interpolation fidelity, and unconditional generation quality.

- **Caution:** straight generative paths z → x do not automatically imply straight physical trajectories s → E(x\*(s)); this remains an empirical hypothesis.

## 5.4 Jacobian and local-isometry diagnostics

Use local Jacobian information primarily as a diagnostic of distortion and conditioning, not as an immediate training objective. A full data-space metric M_X is generally unavailable, and explicit Jacobian regularization may be computationally expensive.

- **Directional sensitivity:** estimate \|\|J_G(z) v\|\| along observed latent trajectory tangents and orthogonal perturbations.

- **Local condition estimates:** compare amplification and contraction across regions of the latent trajectory.

- **Metric sensitivity:** repeat with simple surrogate metrics such as pixel L2 and selected feature-space distances, while treating conclusions as metric dependent.

- **Use:** identify whether interpolation errors arise from a poor latent curve, strong anisotropic decoding, or both.

# 6. Expanded generator suite

A broader suite is needed to distinguish general latent-geometry behavior from properties specific to the bouncing-ball renderer. Generator processes should vary along explicit axes so that failures can be attributed to a known source of complexity.

| **Axis**                 | **Purpose**                                                                               |
|--------------------------|-------------------------------------------------------------------------------------------|
| State dimensionality     | One degree of freedom, two-dimensional motion, or multiple independent factors.           |
| Topology and periodicity | Open interval, circle, product manifold, or repeated phase.                               |
| Dynamics smoothness      | Constant velocity, acceleration, periodic motion, or piecewise-smooth events.             |
| Image formation          | Fully observed, occluded, overlapping, or components entering/leaving the image.          |
| Transformation class     | Rigid translation/rotation, scale, elastic deformation, or distributed dynamics.          |
| Spatial support          | Sparse foreground object versus dense texture or field.                                   |
| Endpoint identifiability | Endpoints uniquely determine the path versus allow multiple plausible paths.              |
| Application similarity   | Video-like temporal evolution versus cross-sectional slice evolution through a 3D object. |

## Recommended generator processes

| **Process**                                      | **Property tested**                                                         | **Priority** |
|--------------------------------------------------|-----------------------------------------------------------------------------|--------------|
| Translated asymmetric object                     | Simple local Euclidean motion and a baseline for exact correspondence.      | High         |
| Rotating arrow or asymmetric shape               | Periodic circular state manifold without translational ambiguity.           | High         |
| Pendulum or orbiting object                      | Periodic motion with nonuniform speed and curved trajectories.              | High         |
| Scaling or smoothly deforming blob               | Appearance change not reducible to rigid motion.                            | High         |
| Two independently moving objects                 | Product structure and partial factor disentanglement.                       | Medium       |
| Crossing objects with occlusion                  | Non-injective image formation and identity preservation.                    | Medium       |
| Bouncing or colliding objects                    | Piecewise-smooth dynamics and sharp changes in tangent.                     | Existing     |
| Slices through ellipsoid, tilted tube, and torus | Direct analogue of sparse volumetric slice interpolation.                   | High         |
| Branching tubular volume                         | Component appearance, disappearance, and topology-sensitive cross sections. | Medium       |
| Advected texture or wave field                   | Dense spatial dynamics beyond sparse foreground objects.                    | Later        |

> **Recommended minimum suite**
>
> Begin with translated asymmetric object, rotating object, smooth deformation, two-object motion, and cross sections through an ellipsoid/tube/torus. Together with the bouncing ball, these cover rigid, periodic, deforming, compositional, event-driven, and volumetric cases.

# 7. Deferred methods and rationale

The following methods are not rejected, but are intentionally deferred because they require direct supervision from dense generator trajectories, simulator state, a trusted data-space metric, or a sequence-specific coupling. The immediate objective is to determine how far representation geometry can be improved with broadly applicable, weakly supervised methods.

| **Deferred method**                        | **Concept**                                                                                   | **Reason for deferral**                                                                                                    |
|--------------------------------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Learned interpolation function or residual | Train a model to predict z\*(s) or a correction to SQUAD.                                     | Requires dense target intermediates and directly learns the desired solution; useful later as an upper bound.              |
| Triplet geodesic loss                      | Force E(x_m) toward SLERP(E(x_a), E(x_b)).                                                    | Directly assumes a desired spherical geometry and uses known intermediate frames; may overconstrain real trajectories.     |
| State-distance preservation                | Match latent distances to distances in simulator state or a physical metric.                  | Requires privileged state variables or a reliable M_X that will not generally exist for real data.                         |
| Sequence-aware or graph-aware coupling     | Assign complete ordered sequences to smooth, correlated latent trajectories.                  | Targets the right property but uses sequence structure in the coupling and is substantially more complex than ordinary OT. |
| Generator-state supervision                | Use known position, velocity, deformation parameters, or topology as latent-geometry targets. | Valuable for controlled studies, but not representative of the information available in microscopy data.                   |

## Possible later use

These methods can later serve as controlled upper bounds. For example, a learned residual interpolator can reveal whether endpoint information is sufficient, and simulator-state metric supervision can reveal the best geometry achievable when the relevant physical variables are known.

# 8. Evaluation framework and success criteria

| **Criterion**              | **Operational interpretation**                                                                    |
|----------------------------|---------------------------------------------------------------------------------------------------|
| Endpoint fidelity          | Observed keyframes reconstruct accurately and remain fixed under interpolation.                   |
| Marginal realism           | Generated frames remain plausible under the learned data distribution.                            |
| Conditional frame accuracy | Intermediate frames match dense reference sequences where available.                              |
| Temporal consistency       | Low frame-to-frame step error, acceleration error, and visual flicker.                            |
| Latent path accuracy       | Low distance to z\*(s), low tangent mismatch, and low subspace residual.                          |
| Prior consistency          | Reasonable radius, mean, standard deviation, and typical-set occupancy.                           |
| Numerical stability        | Convergence under solver refinement and limited sensitivity to batch shape and endpoint handling. |
| Generality                 | Improvements persist across generator axes and keyframe densities.                                |

## Recommended model comparison matrix

- Independent flow matching baseline.

- Optimal-transport-coupled flow matching.

- Independent flow matching plus latent acceleration loss.

- Optimal-transport coupling plus latent acceleration loss.

For each model, evaluate linear interpolation, SLERP, SQUAD, and the hybrid method as a diagnostic. The hybrid should not be treated as the final target method, but it provides evidence about missing correspondence information.

# 9. Current working hypotheses

- **H1.** A pretrained deterministic generative flow can serve as a useful coordinate system for interpolation, provided numerical boundary conditions are handled carefully.

- **H2.** SLERP/SQUAD improve smoothness because they preserve prior typicality and impose coherent latent motion, but they do not guarantee conditional correctness.

- **H3.** The remaining interpolation error is primarily geometric: the true encoded trajectory is not perfectly aligned with the spherical curve implied by independently trained endpoints.

- **H4.** The hybrid improvement indicates that a small correspondence cue can correct part of this mismatch without abandoning the generative prior.

- **H5.** Latent acceleration regularization and optimal-transport coupling are plausible weakly supervised ways to improve trajectory regularity.

- **H6.** Jacobian diagnostics will help distinguish poor latent curves from anisotropic decoder distortion, but are unlikely to be the first practical training objective.

- **H7.** Performance will depend strongly on keyframe density and generator class; conclusions from one bouncing-ball process should not be generalized prematurely.

> **Near-term decision rule**
>
> If the true encoded trajectories have small endpoint-plane or local-subspace residuals, effort should focus on better closed-form interpolation and training smoothness. If those residuals are large, the latent representation itself must be changed; no alternative SLERP weighting can recover directions absent from the endpoint subspace.

# 10. Immediate action list

| **Order** | **Task**                                     | **Output**                                                                                                                              |
|-----------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| 1         | Implement latent-trajectory characterization | SLERP/SQUAD distance, endpoint-plane residual, local subspace residual, radial/angular decomposition, curvature, and tangent alignment. |
| 2         | Run keyframe-density sweeps                  | Measure where local interpolation assumptions begin to fail for each generator process.                                                 |
| 3         | Add generator cases                          | Prioritize translation, rotation, smooth deformation, and simple volumetric cross sections.                                             |
| 4         | Train OT-coupled baselines                   | Hold architecture and compute fixed; compare geometry and generation quality.                                                           |
| 5         | Add a weak latent acceleration loss          | Start locally and at low weight; monitor prior statistics and unconditional sample quality.                                             |
| 6         | Add Jacobian directional diagnostics         | Estimate local contraction/expansion along and orthogonal to observed latent trajectories.                                              |

## Document status

This memo records the current working formulation rather than a final theoretical claim. The central hypothesis remains testable: encode dense observed trajectories, measure their geometry, and determine whether model-training changes make those trajectories more compatible with simple typicality-preserving interpolation in latent space.
