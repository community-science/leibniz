# Latent Factor Complexity

Benchmark complexity is a derived coordinate over declared generator factors,
not a rendering size or a benchmark-specific constant.

Generators expose two related inventories:

- `GeneratorConstructionFactor` records fixed structure of the generator family,
  such as a digit stroke basis, game rules, simulator equations, or physical
  parameter families.
- `SampleLatentFactor` records variables drawn for a concrete sample. Each
  sample factor has a role, initially `content`, `nuisance`, or
  `materialization`.

The primary benchmark coordinate `C` is a `ComplexityProjection` over sample
latent factors. For the first Digits benchmark, `C` is the content projection:
it includes the factors the model is expected to infer and excludes nuisance
and materialization factors. Canvas side length `N` is represented through
resolution requirements, not as the public complexity coordinate.

This keeps the benchmark logic stable if a generator changes its construction.
For example, a digit generator can move from curves to straight lines by
changing its construction-factor declaration while leaving the scoring and
competence projection machinery unchanged.
