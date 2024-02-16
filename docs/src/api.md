## Data Generating Processes

### AbstractDGP
```@docs
AbstractDGP
nfeatures
nparams
priordraw
generate
datatransform
likelihood
ErrorDistribution
```

### Predefined DGPs
```@docs
MA2
Logit
GARCH
```

#### Utilities
```@docs
generate_files
load_from_file
```

## Moment Networks

### HyperParameters
```@docs
HyperParameters
```

### MomentNetwork
```@docs
MomentNetwork
train_network!
apply_transforms
```

## Neural Networks

### Temporal Convolutional Networks
```@docs
TemporalBlock
TCN
```

#### Utilities
```@docs
build_tcn
receptive_field_size
necessary_layers
```

### Recurrent Neural Networks

#### Utilities


### General Neural Network Utilities
```@docs
tabular2conv
tabular2rnn
```