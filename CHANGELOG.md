Changelog
=========

## Changes in version 3

* Changes to the [client code API][1]:
  - relaxed requirements on return type of `configuration()`;
  - require member function `random_configuration()`;
  - require static member function `config_policy_from_parameters()`, a factory
    function for the configuration policy (see below);
  - retired member function `configuration_size()`;
  - retired member types `phase_classifier` and `phase_sweep_policy_type`.
* Split `config_policy` (formerly exclusive to the gauge client code) into an
  inheritance hierarchy of reusable classes:
  - abstract `tksvm::config::policy` interface class;
  - `tksvm::config::monomial_policy` handles index rearrangement, block
    structure as necessary when using the tensorial kernel at given rank;
  - `tksvm::config::clustered_policy` handles calculation of feature mapping for
    a given `ClusterPolicy`, where the latter defines how spin components within
    a chosen spin cluster geometry are retrieved from the raw configuration;
  - a policy that is specific to the client code; can be a type alias to one of
    the former or a derived class.
* The `*-learn` programs have each been split into two separate programs:
  `*-sample` and `*-learn`. The former collects samples and stores these in the
  `*.clone.h5` file, the latter performs the actual SVM optimization.
* Parallelization has been revamped:
  - previous OpenMP-based parallelization has been replaced by one based on MPI;
  - support for [Parallel Tempering][2] (PT);
  - simulation class required to inherit from base class
    `pt_adapter<phase_point>` in lieu of `alps::mcbase` if PT is to be used,
    otherwise from `embarrassing_adapter<phase_point>` for embarassingly
    parallel sampling;
  - uses of the `measurements` object (formerly a protected member inherited
    from `alps::mcbase`) must be replaced by calls to the `measurements()`
    function;
  - only the programs `*-sample` and `*-test` are parallelized.
* Sweep policies:
  - added `nonuniform_grid`, `log_scan`, and `sweep_grid` policies;
  - removed `gaussian_temperatures`, `uniform_temperatures` (previously
    deprecated in version 2).
* Optional "`LAZY`" mode: rather than eagerly mapping raw spin configurations
  sampled from the simulation to the tensorial features and storing those in
  `*.clone.h5`, this may be deferred to the invocation of `*-learn`. This allows
  for a systematic study of multiple different TK ranks and cluster choices from
  a single run of `*-sample` but comes at the cost of having to store the raw
  spin configurations. This behavior can be enabled by setting the
  [build option][3] `CONFIG_MAPPING` to `LAZY`. The old behavior, "`EAGER`",
  remains the default.
* The [classifier][4] is no longer chosen at compile time using a build option,
  but at run time using the `classifier.policy` parameter.
* The `fixed_from_sweep` classifier replaces both `fixed_from_cycle` and
  `fixed_from_grid` and is applicable to arbitrary sweep policies.
* New client code [`frustmag`][5]: highly generic code for classical MC
  simulations of frustrated spin systems.
* Features added to `*-learn` program:
  - flag `--infinite-temperature` allows for inclusion of fictious spin
    configuration samples corresponding to completely uncorrelated "infinite
    temperature" spins (obtained via the `random_configuration()` function) as a
    control group;
  - flag `--statistics-only` applies the classifier, prints label statistics,
    but foregoes the SVM optimization;
  - flag `--skip-sampling` removed, as `*-learn` no longer samples;
  - flag `--merge` allows to combine the samples of multiple `*.clone.h5` files
    into one big optimization problem.
* Features added to `*-test` program:
  - The set of points at which the SVM decisions function(s) and observables are
    to be measured is no longer constrained to a simple line in parameter space
    but can be specified in the form of a sweep policy using parameters
    `test.policy` and the like.
  - program can be launched with an `*.ini` parameter file, in which case no SVM
    decision functions are measured, only the observables registered and
    measured by the simulation class;
  - calculation and output of variances on decision functions and observables
    (provided also their square is measured);
  - `*.test.txt` output files are self-documenting: a comment annotating their
    column layout is automatically generated.
* Changes to the `*-coeffs` program:
  - `--exact` and `--diff` are now conditional on the presence of the
    `--result=<result-name>` flag, specifying the name of the (hardcoded)
    reference result that is to be compared to.
* Features added to the graph analysis (`*-segregate-phases`):
  - binary decision on whether or not to include an edge in the graph is
    superseded by an option `--weight` to specify a weighting function which
    maps biases to continuous edge weights in [0, 1]; default value `box`
    reproduces previous behavior;
  - multiple graphs may be combined into one (by multiplication of edge weights)
  - support for masks (`--threshold`, `--invert-mask`, `--mask`,
    `--masked-value`)


## Changes in version 2

* Support for multiclassification
  - in upstream SVM wrapper library
  - all decision functions are being measured in `*-test`
* Gauge client code uses both _J<sub>1</sub>_ and _J<sub>2</sub>_ as parameters
  which locate the model in the phase diagram
  - introduced _sweep policy_ concept to traverse phase diagram in varied ways
  - introduced _classifier_ concept to map points in the phase diagram to labels
    which are used as input for the SVM
* Spectral graph partitioning analysis using separate program:
  `*-segregate-phases`


[1]: ./README.md#client-code-api
[2]: ./README.md#parallel-tempering
[3]: ./README.md#build-configuration
[4]: ./README.md#classifiers
[5]: ./frustmag/README.md
