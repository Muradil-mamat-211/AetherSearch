# Configurations

The supported public entry configuration is
`recipes/rl/train_4x48gb.yaml`. It composes the verified algorithm settings in
this directory with `configs/hardware/4x48gb_3rl.yaml`, while all machine-local
paths come from `environment/env.local.sh`.

`formal_resolved/` contains immutable execution records from the released
training chain. Those files intentionally preserve the original absolute paths
and launch commands for provenance; they are records, not portable templates.

The other root YAML files are inherited algorithm, retriever, schedule, gate,
and historical qualification layers. Public launches should use the recipe,
not invoke those layers independently.
