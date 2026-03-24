# TODO

## 25.01.15
- [x] Folder restructuring
- [x] Proper configuration
    - [x] PyYAML
    - [x] Pydantic?
        - [x] [Replace your YAML configs with Pydantic](https://www.youtube.com/watch?v=4JAdedd_G-w)
            * Type safety & Validation
            * Dynamic Configuration
            * Domain Model

## 25.01.17
- [ ] Migrate training on to NSCC's HPCs
  * $100k grant
  - [ ] [NSCC ASPIRE2A: A Beginner's Guide to Running AI Jobs](https://help.nscc.sg/wp-content/uploads/2024/05/ASP2A-AI-Guide-7_March_2024.pdf)
  - [ ] [NSCC Training on AI and ML](https://youtu.be/x_ArUs_UMxw?si=nqbBzTPKGVLK8p1c)
  - [ ] [How to run batch jobs on NSCC HPC](https://nusit.nus.edu.sg/services/getting-started/how-to-run-batch-job/)

- [ ] ~~Tensorboard Logging service~~

- [x] Inference server
  - [x] [Litserve](https://lightning.ai/blog/evolution-of-model-inference#start-simple)
    - [x] Pydantic Request/Response setup
  - [x] Streamlit
  * To run API servers: 
    ```python -m apis```
  * To run inference dashboard: 
    * ~~```streamlit app/[app-name]/app.py```~~
    * ```streamlit home.py```

### 25.01.18
- [x] (?) Update loss function to account for image derivative (smoothness)

### 25.01.21
- [x] Integration with Wandb logger
  - [?] Consider Pro for Academics

- [ ] Setup dataset: https://github.com/Spiideo/sskit?tab=readme-ov-file
  - [x] COCO object (translate to Pydantic model?)

## 25.02.04
- [x] Refactoring LitServe API servers
  * Always inherit Mixin methods before the base class, due to Python's Multiple Resolution Order (MRO)

## 25.02.09. Meeting with Prof Misha
- [x] To send Prof:
  - [x]  Plan/Skeleton for the model
  - [x] Information on the IMS summer research program

## 25.03.07. Implement ~~YOLO~~ Gaussian Heatmap
  - [x] Installed sskit
  - ~~[ ] YOLOx Model~~
  - [x] Dataset
  - [x] Experiment

## 25.03.22. Model tuning
  - Losses:
    - [x] Focal Loss
    - [x] MSE Loss
    - [x] TotalVariation Loss
    - [ ] Comparing the derivatives/Laplacian?
    - [ ] Fourier/Spectral Loss

  - Pipeline:
    - [?] Gaussian Mixture Model
    - [?] Bayesian Gaussian Mixture Model

  - Model considerations:
    - UNet heatmap dimensions?
      - [?] 512 x 512
      - [ ] 240 x 240

    - Dataset
      - Take n/2 crops with the most objects and n/2 crops with the least objects. (1:1)