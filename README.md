# Focal Loss
This is the keras implementation of focal loss proposed by [Lin](https://vision.cornell.edu/se3/people/tsung-yi-lin/) et. al. in their [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) paper.
# Usage
You have to compile your model with focal loss.
Sample:
```
model_prn.compile(optimizer=optimizer, loss=[focal_loss(alpha=2, gamma=2)])
```

