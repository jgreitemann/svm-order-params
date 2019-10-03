checkpointing_stop_callback.hpp
-------------------------------
checkpointing_stop_callback

config_policy.hpp
-----------------
element_policy::components
cluster_policy::stride
symmetry_policy::none
symmetry_policy::symmetrized
block_reduction::inf
block_reduction::norm
block_reduction::sum
config_policy
monomial_config_policy
clustered_config_policy
dummy_inspector
block_config_policy

config_serialization.hpp
------------------------
config_serializer

config_sim_base.hpp
-------------------
(Macros)

contraction.hpp
---------------
contraction

dispatcher.hpp
--------------
dispatcher

embarrassing_adapter.hpp
------------------------
embarrassing_adapter

phase_space_point.hpp
---------------------
phase_space::point::temperature
phase_space::point::J1J3
phase_space::point::JpmT
phase_space::point::distance

phase_space_policy.hpp
----------------------
phase_space::label::numeric_label
phase_space::sweep::define_parameters
phase_space::sweep::from_parameters
phase_space::sweep::policy
phase_space::sweep::cycle
phase_space::sweep::grid
phase_space::sweep::nonuniform_grid
phase_space::sweep::sweep_grid
phase_space::sweep::uniform
phase_space::sweep::uniform_line
phase_space::sweep::line_scan
phase_space::sweep::log_scan
phase_space::classifier::policy
phase_space::classifier::critical_temperature
phase_space::classifier::orthants
phase_space::classifier::hyperplane
phase_space::classifier::phase_diagram
phase_space::classifier::phase_diagram_database
phase_space::classifier::fixed_from_sweep
phase_space::classifier::define_parameters
phase_space::classifier::from_parameters

procrastination_adapter.hpp
---------------------------
procrastination_adapter

pt_adapter.hpp
--------------
iso_batcher
pt_adapter

results.hpp
-----------
results::index_rule
results::delta_rule
results::make_delta
results::distinct_rule
results::make_distinct
results::contraction
results::tensor_factory
results::exact_tensor

test_adapter.hpp
----------------
test_adapter

training_adapter.hpp
--------------------
training_adapter

frustmag/concepts.hpp
---------------------
...

frustmag/frustmag.hpp
---------------------
frustmag_sim

frustmag/frustmag_config_policy.hpp
-----------------------------------
element_policy::lattice
element_policy::single
cluster_policy::frustmag_single
cluster_policy::frustmag_lattice
frustmag_config_policy
define_frustmag_config_policy_parameters
frustmag_config_policy_from_parameters

frustmag/observables.hpp
------------------------
obs::energy
obs::magnetization
...
observables

frustmag/std_concepts.hpp
-------------------------
...

frustmag/gauge.hpp
------------------
gauge_sim
config_serializer

frustmag/gauge_config_policy.hpp
--------------------------------
element_policy::mono
element_policy::triad
element_policy::n_partite
cluster_policy::single
cluster_policy::square
cluster_policy::full
gauge_config_policy
define_gauge_config_policy_parameters
gauge_config_policy_from_parameters

utilities/filesystem.hpp
------------------------
replace_extension

utilities/indices.hpp
---------------------
indices_t
block_indices_t
...

utilities/matrix_output.hpp
---------------------------
normalize_matrix
write_matrix

utilities/mpi.hpp
-----------------
mpi::*

utilities/polygon.hpp
---------------------
polygon