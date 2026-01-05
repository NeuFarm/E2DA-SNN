<div align="center">

## Event-Driven Dual-Attention Spiking Neural Network  
## for Energy-Efficient Detection of Coffee Fruit Maturity
</div>

This repository provides the official implementation of **E2DA-SNN**,  
an event-driven spiking neural network for energy-efficient object detection.  

The code is released to support reproducibility of the experimental results reported in the paper.

---

### Requirements

The code has been tested with the following environment:

- Python: 3.8  
- PyTorch: 1.10.0  
- CUDA: 11.3  
- cuDNN: 8.2.0  

<details open>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>

### Pretrained Checkpoints

We provide pretrained [E2DA](https://drive.google.com/file/d/1VsOd_sk0Wf6R8SFKw9sLLP9xopR7hcHw/view?usp=sharing) and [E2DA-Lite](https://drive.google.com/file/d/1Kmgp-MgIiW2n2igNH68Xw-21OB3YuJxx/view?usp=sharing) models on the COFFEE_FOB dataset.

### Training 
<details open>
<summary>Train</summary>

Train the standard E2DA model:
```bash
python train.py --cfg models/e2da.yaml --data data.yaml --weights path/to/weights.pt
```
Train the lightweight E2DA-Lite model:
```bash
python train.py --cfg models/e2da_lite.yaml --data data.yaml --weights path/to/weights.pt
```

</details>

### Evaluation
<details open> <summary>Validate / Evaluate Models</summary>

Evaluate a trained model:
```bash
python val.py --weights path/to/weights.pt --data data.yaml
```

</details>
