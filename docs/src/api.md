## Data Generating Processes

### AbstractDGP
```@docs
AbstractDGP
nfeatures
nparams
priordraw
generate
likelihood
```

### Predefined DGPs
```@docs
MA2
Logit
GARCH
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