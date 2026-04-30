= An Incremental Development Pipeline

#figure(
  caption: [The project's current development pipeline],
)[
  #image("../infra/infra.svg")
]

As the project grows with different tasks and diverse sources of datasets, the pure notebook approach in ISC-1 proves to be insufficient. Whilst the project already has seperate modules for calling certain dataset (original or augmented), it still lacks a reproducible training pipeline. Moreover, each model requires multiple, possibily overlapping, steps of input preprocessing, therefore the preprocessing scripts must also be seperated as a designed class for reusability.

To refactor the project codebase, we follow a selection of best practices from Imperical College London's _Deep Learning Best Practices_ @noauthor_recode-deeplearning-best-practices_nodate. Due to time constraints and because of the already sophisicated codebase from the first ISC, we only adapt certain suggestions, whilst skipping on some libraries such as _Hydra_ or _Einops_. These are the possible points for improvement moving forward when another code refactoring is needed.

#figure(
  placement: top,
  caption: [Personal reflection summary on the six _'Deep Learning best practices'_ @noauthor_recode-deeplearning-best-practices_nodate by Imperial College London],
)[
  #table(
    columns: (auto, 10%, auto),
    align: horizon,
    [*Tools*], [*Usage*], [*Intended/Alternative use*],
    [_Wandb_, metrics and artifacts logging], [🟢], [],
    [_Hydra_, manage configurations],
    [🟡],
    [
      *Alternative*: Uses `config.yaml` for experiments
    ],
    [_Pytorch Lightning_, training models], [🟢], [Supports multicore GPU use, intergratability with cloud GPU usage],
    [_Einops_, tensor multiplcation], [🟡], [Interpretable tensor manipulation],
    [Training on GPU], [🟢], [Supported via _Pytorch Lightning_],
    [Project template], [🟢], [],
    table.cell(colspan: 3)[
      #set align(right)
      #set text(7pt)
      _
      🟢: adapted,
      🟡: yet to be adapted
      _
    ],
  )
]


After careful consideration, PyTorch Lightning @noauthor_welcome_nodate is chosen because of its modularity and extendability onto the cloud.

- *Modularity*. Each "module" in the project can be expressed as a `pl.LightningModule` object. The modules are equipped with prewritten scripts to interact with `pl.Trainer`s (for training) and loggers, whilst being flexible as the developer can customise the module via one of its functions in the `pl.LightningModule` API.

- *Compability with on-device and on-demand GPUs*. Training experiments are automatically orchestrated via `pl.Trainer`. It handles the usage of GPUs on the background and communicates with the `DataModules` to load batched input data and with the loggers to output the necessary artifacts and weights.

It is important that the project needs to be flexible because the datasets, models and pipelines are prone to methodological and technological changes in the future. We shall adapt ICL's @noauthor_recode-deeplearning-best-practices_nodate project structure, with some add-ons to the modules to fit the needs of the project.

Our project consists of isolated modules:

- *`data`*: contains raw data, including images (in `.jpg` format) and `JSON` annotation files.
- *`datamodules`*: contains `pl.LightningDataModule`s that are responsible of reading input images from the *`data`* folder, and perform any transformation on the images before feeding for model training.
- *`models`*: contains trainable `pl.LightningModule`s.
- *`pipeline`*: contains non-trainable `pl.LightningModule`s, only for evaluation. These are mainly composition modules that connects trainable models with non-trainable layers to perform a full prediction pipeline.
- *`losses`*: contains non-trainable `pl.LightningModule`s to determine the loss function between $y_"pred"$ and $y_"true"$.
- *`callbacks`*: contains callback functions that are utilised by the logger to log model artifacts, which will become useful for later model diagnostics.

These modules are then orchestrated by experiment scripts in the *`experiments`* folder. These are where the training happens, with configuration parameters determined within each individual experiment via the `config.yaml` file.
